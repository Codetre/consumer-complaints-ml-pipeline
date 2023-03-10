import absl
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

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


context = InteractiveContext()

# ??????????????? ????????? ?????? ?????????

# `_`??? ???????????? ???????????? ???????????? ??? ???????????? ?????????????????? ???????????? ???????????????.
# ????????? ?????? ????????? ???????????? ??????, ????????? ?????? ???????????? ??? ??????????????? ??????.
_pipeline_root = "pipeline_root"
_pipeline_name = "consumer_complaints"
_metadata_path = os.path.join(_pipeline_root, "metadata.sqlite")  # Default for MLMD is SQLite.

# ???????????? ??????????????? ????????? ?????? ?????????
data_dir = "data/complaints/records"
serving_model_dir = os.path.join(_pipeline_root, "serving_models")
modules = {
    "trainer": "trainer_module.py",
    "transform": "transform_module.py"}

notebook_filepath = os.path.join(os.getcwd(), "notebooks_for_practice/pipeline_exporting.ipynb")

components = init_components(data_dir=data_dir,
                             modules=modules,
                             serving_model_dir=serving_model_dir)

# `_beam_pipeline_args = [f'--requirements_file={path}']`??? ?????? ??????
requirements_file = "requirements.txt"
export_filepath = "beam_pipeline.py"
runner_type = "beam"

# ????????? ?????? ????????? ????????? ???????????? ??????????????? ??????????????? ????????????.
# ??? ?????????????????? ??????????????? ???????????? ????????? ??????????????? ???????????? ????????? ?????????????????? ????????????.
context.export_to_pipeline(notebook_filepath=notebook_filepath,
                           export_filepath=export_filepath,
                           runner_type=runner_type)

absl.logging.set_verbosity(absl.logging.INFO)

# Pipeline args for Beam jobs within Components.
_beam_pipeline_args = [
    '--direct_running_mode=multi_processing',
    # 0 means auto-detect based on on the number of CPUs available
    # during execution time.
    '--direct_num_workers=0',
]

tfx_pipeline = pipeline.Pipeline(
    pipeline_name=_pipeline_name,
    pipeline_root=_pipeline_root,
    components=components,
    enable_cache=True,
    metadata_connection_config=(
        metadata.sqlite_metadata_connection_config(_metadata_path)),
    beam_pipeline_args=_beam_pipeline_args,
    additional_pipeline_args={})

BeamDagRunner().run(tfx_pipeline)