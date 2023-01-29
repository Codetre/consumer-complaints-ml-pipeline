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


def transform_name(key: str):
    return key + "_xf"


def fill_in_missing(x):
    if isinstance(x, tf.sparse.SparseTensor):
        default_value = "" if x.dtype == tf.string else 0
        dense_shape = [x.dense_shape[0], 1]
        x = tf.sparse.to_dense(
            # `tf.sparse.SparseTensor`의 별칭.
            tf.SparseTensor(indices=x.indices,  # non-0 위치.
                            values=x.values,  # `indices`에 지정된 위치에 들어갈 값들.
                            dense_shape=dense_shape),
            default_value=default_value
        )
    return tf.squeeze(x, axis=1)  # 크기 1인 차원을 없앤다. `axis`로 축소할 차원 나열.


def convert_num_to_one_hot(label_tensor, num_labels=2):
    one_hot_tensor = tf.one_hot(
        indices=label_tensor,
        depth=num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])  # not-fixed-size batch


def convert_zip_code(zip_code):
    if zip_code == "":
        zip_code = "00000"

    zip_code = tf.strings.regex_replace(zip_code, r"X{0,5}", "0")
    zip_code = tf.strings.to_number(zip_code, out_type=tf.int64)
    return zip_code


def preprocessing_fn(inputs):
    outputs = {}

    for key in ONE_HOT_FEATURES.keys():
        dim = ONE_HOT_FEATURES[key]
        idx = tft.compute_and_apply_vocabulary(
            fill_in_missing(inputs[key]), top_k=dim + 1
        )
        outputs[transform_name(key)] = convert_num_to_one_hot(
            idx, num_labels=dim + 1
        )

    for key, bucket in BUCKET_FEATURES.items():
        temp_feature = tft.bucketize(
            fill_in_missing(inputs[key]),
            bucket,
            # always_return_num_quantiles=False
        )

        outputs[transform_name(key)] = convert_num_to_one_hot(
            temp_feature, num_labels=bucket + 1
        )

        for key in TEXT_FEATURES.keys():
            outputs[transform_name(key)] = fill_in_missing(inputs[key])

        outputs[transform_name(LABEL_KEY)] = fill_in_missing(inputs[LABEL_KEY])

    return outputs
