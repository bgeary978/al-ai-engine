from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    sqm = data.get("sqm")
    bedrooms = data.get("bedrooms")
    county = data.get("county")

    print(f"Received: {sqm} sqm, {bedrooms} bedrooms, in {county}")

    return jsonify({"estimated_value": 350000})

if __name__ == "__main__":
    app.run(debug=True)