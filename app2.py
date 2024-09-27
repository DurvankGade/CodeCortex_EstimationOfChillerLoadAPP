from fastapi import FastAPI, File, UploadFile
import pandas as pd
import uvicorn
import pickle
from io import BytesIO
from fastapi import FastAPI, File, UploadFile
from starlette.responses import StreamingResponse
app = FastAPI()

# Load the pickled model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


@app.post("/predict")
async def predict_chiller_frequency(csv_file: UploadFile = File(...)):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file.file)

    # Ensure the DataFrame has the correct columns for your model
    if not all(col in df.columns for col in model.feature_names_in_):
        raise ValueError("CSV file must contain columns: {}".format(model.feature_names_in_))

    # Make predictions using the model
    predictions = model.predict(df)

    # Add "chiller_frequency" column to the DataFrame
    df["chiller_frequency"] = predictions.tolist()

    # Create a temporary in-memory file object
    in_memory_file = BytesIO()

    # Save the modified DataFrame back to the CSV format
    df.to_csv(in_memory_file, index=False)

    # Reset the in-memory file stream to the beginning
    in_memory_file.seek(0)

    # Return the updated CSV data as a download
    return StreamingResponse(
        content=in_memory_file,
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=updated_{csv_file.filename}"
        }
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)