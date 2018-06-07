class Client:
    def request(self, input_seq):
        raise NotImplementedError()

    def request_many(self, input_seqs)
        raise NotImplementedError()


class GNMTClient(Client):

    def __init__(self, model_name="address", host="localhost", port=9000, timeout=10):
        self.model_name = model_name
        self.host = host
        self.port = port
        self.timeout = timeout
        channel = implementations.insecure_channel(self.host, self.port)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    def request(self, input_seq):
        future = self._translate(input_seq)
        result = self._parse_result(future)
        return (input_seq, result)

    def request_many(self, input_seqs):
        futures = []
        for s in input_seqs:
            future = self._translate(s)
            futures.append(future)
        pairs = []
        for seq, future in zip(input_seqs, futures):
            result = self._parse_result(future)
            pairs.append((seq, result))
        return pairs

    def _parse_result(self, future):
        result = self._parse_translation(future.result())
        words = ""
        for w in list(result):
            words += str(w, encoding="utf8") + " "
        return words

    def _translate(self, seq):
        request = predict_pb2.PredictRequest()
        # model_name should keep the same as tf serving start arg `--model_name`
        request.model_spec.name = self.model_name
        # signature_name should keep the same as your `signature_def_map` 's `key` in `exporter`
        request.model_spec.signature_name = "serving_default"
        # `seq_input` should be the same as the `inference_signature` in the `exporter`
        request.inputs["seq_input"].CopyFrom(tf.make_tensor_proto(seq, dtype=tf.string, shape=[1, ]))
        return self.stub.Predict.future(request, self.timeout)

    @staticmethod
    def _parse_translation(result):
        # `seq_output` should be the same as the `inference_signature` in the `exporter`
        inference_output = tf.make_ndarray(result.outputs["seq_output"])
        return inference_output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True, help="model name")
    parser.add_argument("--host", default="localhost", help="model server host")
    parser.add_argument("--port", type=int, default=9000, help="model server port")
    parser.add_argument("--timeout", type=float, default=10.0, help="request timeout")
    args = parser.parse_args()

    test_seqs = [
        "上海 浦东新区 张东路",
        "浙江 杭州 下沙区",
        "北京市　海淀区　北京西路"
    ]
    client = GNMTClient(model_name=args.model_name, host=args.host, port=args.port, timeout=args.timeout)

    input_seq, output_seq = client.request(test_seqs[0])
    print("Input : %s" % input_seq)
    print("Output: %s" % output_seq)
   
    results = client.request_many(test_seqs)
    for r in results:
        print("Input : %s" % r[0])
        print("Output: %s" % r[1])
