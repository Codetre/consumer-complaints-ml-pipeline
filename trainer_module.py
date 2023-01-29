import os

import tensorflow as tf
import tensorflow_transform as tft

# 피처명, 차원(가능한 값의 수)
ONE_HOT_FEATURES = {
    "product": 11,
    "sub_product": 45,
    "company_response": 5,
    "state": 60,
    "issue": 90,
}

# 피처명, 버킷 수(범위 수)
BUCKET_FEATURES = {
    "zip_code": 10
}

# 피처명, 미정의
TEXT_FEATURES = {
    "consumer_complaint_narrative": None
}

LABEL_KEY = "consumer_disputed"


def transform_name(key):
    return key + "_xf"


def _gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, batch_size=32):
    # 미리 변환된 데이터셋을 불러오도록 하려면 피처를 기술한 feature spec이 필요하다.
    # 대신 그 경우 변환과 추론 그래프가 통합된 모델 내보내기란 장점을 포기해야 한다.
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    # `make_batched_features_dataset`는 배치별 데이터 제공 제너레이터를 반환한다.
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        label_key=transform_name(LABEL_KEY)
    )

    # 이 반환 변수는 (features, indices) tuple을 포함하며, features는 Tensor들의 dict,
    # indices는 라벨을 의미하는 Tensor다.
    return dataset


def get_model():
    import tensorflow_hub as hub

    input_features = []
    # 피처를 루프 돌리며 각 피처에 대한 input_feature를 작성합니다.
    for key, dim in ONE_HOT_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,),
                           name=transform_name(key)))

    # 버킷화 피처를 추가
    for key, dim in BUCKET_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,),
                           name=transform_name(key)))

    # 문자열 피처를 추가
    input_texts = []
    for key in TEXT_FEATURES.keys():
        input_texts.append(
            tf.keras.Input(shape=(1,),
                           name=transform_name(key),
                           dtype=tf.string))

    inputs = input_features + input_texts

    # 문자열 피처를 임베딩
    MODULE_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"
    # 범용 문장 인코더 모델의 tf.hub 모듈을 로드합니다.
    embed = hub.KerasLayer(MODULE_URL)
    # 케라스 입력은 2차원이지만 인코더는 1차원 입력을 기대합니다.
    reshaped_narrative = tf.reshape(input_texts[0], [-1])
    embed_narrative = embed(reshaped_narrative)
    deep_ff = tf.keras.layers.Reshape((512,), input_shape=(1, 512))(embed_narrative)

    deep = tf.keras.layers.Dense(256, activation='relu')(deep_ff)
    deep = tf.keras.layers.Dense(64, activation='relu')(deep)
    deep = tf.keras.layers.Dense(16, activation='relu')(deep)

    wide_ff = tf.keras.layers.concatenate(input_features)
    wide = tf.keras.layers.Dense(16, activation='relu')(wide_ff)

    both = tf.keras.layers.concatenate([deep, wide])

    output = tf.keras.layers.Dense(1, activation='sigmoid')(both)
    # 피처 API로 모델 그래프를 조립합니다.
    keras_model = tf.keras.models.Model(inputs, output)

    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=[
                            tf.keras.metrics.BinaryAccuracy(),
                            tf.keras.metrics.TruePositives()
                        ])
    return keras_model


# 이 부분을 출력 시그니처(`serving_default`)에 매핑함으로써 변환 단계를 모델에 병합하게 된다.
def _get_serve_tf_examples_fn(model, tf_transform_output):
    # 변환 그래프 로드.
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        # 변환 전 직렬화된 원시 데이터를 피처 사양으로 구조화.
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        # 원시 데이터에 변환 적용.
        transformed_features = model.tft_layer(parsed_features)
        # 변환된 데이터로 예측.
        outputs = model(transformed_features)

        return {"outputs": outputs}

    return serve_tf_examples_fn


def run_fn(fn_args):
    # 1. 입력 데이터 준비
    # `fn_args.transform_output`: `Transform`을 거쳐 나온 변환 그래프, 예제 데이터셋,
    # 학습 매개변수 등
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_fn(fn_args.train_files, tf_transform_output)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output)

    # Tensorboard 모니터링을 위한 준비
    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "log")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                          update_freq="batch")
    # 2. 모델 빌드 및 컴파일
    model = get_model()

    # 3. 모델 훈련
    # 분산 처리의 경우는 아래와 같이 진행한다
    # (`MirroredStrategy`는 단일 인스턴스 복수 GPU일 때, 각 GPU에 모델과 파라미터를 동일하게
    # 부여하고 서로 다른 배치로 훈련시킨 후 각 결과를 동기식-모든 레플리카의 그래디언트 계산이
    # 모델 업데이트 전에 끝난다-으로 종합한다).
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    #     model.fit()

    # epochs==n으로 학습하려면 steps=(num_batches * n)으로 설정하면 된다.
    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback]
    )

    # 4. 학습된 모델 내보내기
    signatures = {
        "serving_default": _get_serve_tf_examples_fn(
            model, tf_transform_output).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        )
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
