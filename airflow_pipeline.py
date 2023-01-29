"""Complaints model DAG

Command for creating Airflow admin
`airflow users  create --role Admin  \
    --username admin --password XXXX \
    --email my@email.com --firstname ML --lastname Pipeline`

이 파일은 $AIRFLOW_HOME/dags 아래 저장해야 한다.
`python this.py` 실행으로 그래프 등록 후, `airflow scheduler`와 `airflow webserver`로
스케줄러와 웹 서버를 가동하면 된다.

"""
import absl
import datetime
from tfx.orchestration.airflow import airflow_dag_runner

import os
from datetime import datetime

import tensorflow_model_analysis as tfma
from tfx.components import (ImportExampleGen, StatisticsGen, SchemaGen,
                            ExampleValidator, Transform, Trainer, Evaluator,
                            Pusher)
from tfx.dsl.components.common.resolver import Resolver
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.proto import pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.v1.dsl.experimental import LatestBlessedModelStrategy
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

def init_components(data_dir,
                    modules: dict,
                    serving_model_dir,
                    training_steps=100,
                    eval_steps=50):
    # Config.
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="consumer_disputed")],
        slicing_specs=[tfma.SlicingSpec(), tfma.SlicingSpec(feature_keys=["product"])],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="BinaryAccuracy"),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(class_name="AUC")],
                thresholds={
                    "AUC": tfma.MetricThreshold(
                        value_threshold=tfma.GenericValueThreshold(
                            lower_bound={"value": 0.65}),
                        change_threshold=tfma.GenericChangeThreshold(
                            direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                            absolute={"value": 0.01}))})])

    # Components
    # 1. Example generation
    example_gen = ImportExampleGen(input_base=data_dir)
    examples = example_gen.outputs["examples"]
    print("1. Example gathering finished.")

    # Get example statistics
    statistics_gen = StatisticsGen(examples=examples)
    stats = statistics_gen.outputs["statistics"]
    print("2. Statistics calculation finished.")

    # Get example schema
    schema_gen = SchemaGen(statistics=stats)
    schema = schema_gen.outputs["schema"]
    print("3. Schema generation finished.")

    # Example validation
    example_validator = ExampleValidator(statistics=stats, schema=schema)
    anomalies = example_validator.outputs["anomalies"]
    print("4. Example validation finished.")

    # Data transform
    transform = Transform(examples=examples,
                          schema=schema,
                          module_file=modules['transform'])
    transformed_examples = transform.outputs["transformed_examples"]
    transform_graph = transform.outputs["transform_graph"]
    print("5. Data transformation finished.")

    # Model training
    trainer = Trainer(examples=examples,
                      transform_graph=transform_graph,
                      schema=schema,
                      module_file=modules["trainer"],
                      train_args=trainer_pb2.TrainArgs(num_steps=training_steps),
                      eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps))
    model = trainer.outputs["model"]
    print("6. Model training finished.")

    # Model resolving(choose one among conflicting models.)
    resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing)
    ).with_id("latest_blessed_model_resolver")
    print("7. Model resolving finished.")

    # Model evaluation
    evaluator = Evaluator(
        examples=examples,
        model=model,
        baseline_model=resolver.outputs["model"],
        eval_config=eval_config)
    blessing = evaluator.outputs["blessing"]
    evaluation = evaluator.outputs["evaluation"]
    print("8. Model evaluation finished.")

    # Model pushing
    push_filesystem = pusher_pb2.PushDestination.Filesystem(base_directory=serving_model_dir)
    push_destination = pusher_pb2.PushDestination(filesystem=push_filesystem)
    pusher = Pusher(model=model,
                    model_blessing=blessing,
                    push_destination=push_destination)
    print("9. Model pushing finished.")

    components = [example_gen, statistics_gen, schema_gen, example_validator,
                  transform, trainer, resolver, evaluator, pusher]
    return components


# The trace that explains this script comes from Jupyter notebook. The
# interactive context manager doesn't orchestrate the pipeline, the airflow does
# it.
context = InteractiveContext()

# 파이프라인 정의를 위한 인자들

# `_`로 시작하는 경로들은 내보내기 시 생성되는 스크립트에서 요구하는 경로명이다.
# 바꾸지 말고 그대로 사용해야 하며, 빼놓지 말고 정의해야 할 것들이기도 하다.
_pipeline_root = "pipeline_root"
_pipeline_name = "consumer_complaints"
_metadata_path = os.path.join(_pipeline_root, "metadata.sqlite")  # Default for MLMD is SQLite.

# 컴포넌트 인스턴스들 생성을 위한 인자들
project_dir = "/home/hakjun/projects/pipeline"
data_dir = os.path.join(project_dir, "data/complaints/records")
serving_model_dir = os.path.join(_pipeline_root, "serving_models")
modules = {
    "trainer": os.path.join(project_dir, "trainer_module.py"),
    "transform": os.path.join(project_dir, "transform_module.py")}

notebook_filepath = os.path.join(os.getcwd(), "notebooks_for_practice/airflow_exporting.ipynb")

components = init_components(data_dir=data_dir,
                             modules=modules,
                             serving_model_dir=serving_model_dir)

# Airflow-specific configs; these will be passed directly to airflow
_airflow_config = {
    "schedule_interval": None,
    "start_date": datetime.today()}

export_filepath = "airflow_pipeline.py"
runner_type = "airflow"

context.export_to_pipeline(notebook_filepath=notebook_filepath,
                           export_filepath=export_filepath,
                           runner_type=runner_type)



# Pipeline args for Beam jobs within Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

absl.logging.set_verbosity(absl.logging.INFO)

tfx_pipeline = pipeline.Pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    components=components,
    enable_cache=True,
    metadata_connection_config=(
        metadata.sqlite_metadata_connection_config(_metadata_path)),
    beam_pipeline_args=_beam_pipeline_args,
    additional_pipeline_args={})

# 'DAG' below needs to be kept for Airflow to detect dag.
DAG = airflow_dag_runner.AirflowDagRunner(
    airflow_dag_runner.AirflowPipelineConfig(_airflow_config)).run(
      tfx_pipeline)