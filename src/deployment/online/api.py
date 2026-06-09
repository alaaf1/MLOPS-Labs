import pickle
import pandas as pd
import litserve as ls
from src.deployment.online.requests import InferenceRequest

class TitanicInferenceAPI(ls.LitAPI):
    
    def setup(self, device="cpu"):
        # load both models
        with open("models/random_forest.pkl", "rb") as f:
            self._rf_model = pickle.load(f)
        with open("models/logistic_regression.pkl", "rb") as f:
            self._lr_model = pickle.load(f)
        # default to random forest (better CV AUC)
        self._model = self._rf_model

    def decode_request(self, request):
        try:
            # validate input using pydantic
            parsed = InferenceRequest(**request)
            # convert to dataframe — pipeline expects this
            records = [p.model_dump() for p in parsed.inputs]
            return pd.DataFrame(records), None
        except Exception as e:
            return None, str(e)

    def predict(self, inputs):
        df, error = inputs
        if df is None:
            return None, error
        predictions = self._model.predict(df).tolist()
        probabilities = self._model.predict_proba(df)[:, 1].tolist()
        return predictions, probabilities

    def encode_response(self, output):
        predictions, probabilities_or_error = output

        if predictions is None:
            return {
                "message": "Error occurred",
                "error": probabilities_or_error,
                "predictions": []
            }

        results = []
        for pred, prob in zip(predictions, probabilities_or_error):
            results.append({
                "survived": bool(pred),
                "survival_probability": round(prob, 4)
            })

        return {
            "message": "Predictions generated successfully",
            "predictions": results
        }