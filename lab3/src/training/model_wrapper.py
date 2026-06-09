from mlflow.pyfunc import PythonModel, PythonModelContext


class ModelWrapper(PythonModel):
    def load_context(self, context: PythonModelContext):
        import pickle
        with open(context.artifacts["encoder"], "rb") as f:
            self._encoder = pickle.load(f)
        with open(context.artifacts["model"], "rb") as f:
            self._model = pickle.load(f)

    def predict(self, context: PythonModelContext, model_input):
        import numpy as np
        preds = self._model.predict(model_input)
        if isinstance(preds, np.ndarray):
            preds = preds.tolist()
        return [self._encoder["decoder"][int(val)] for val in preds]