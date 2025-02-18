# Variables
PYTHON=python
PIP=pip
VENV=.venv
REQ=requirements.txt
TRAINED_MODEL=trained_model.joblib
SCALER=scaler.joblib
PREPROCESSED_DATA=preprocessed_data.joblib
DATA_FILE=/home/amor/ml_project/churn-bigml-80.csv
TARGET_COLUMN=Churn

# Create a virtual environment and install dependencies
install:
	@echo "Creating virtual environment..."
	${PYTHON} -m venv ${VENV}
	@echo "Activating virtual environment and installing dependencies..."
	${VENV}/bin/${PIP} install -r ${REQ}

# Format and lint the code
lint:
	@echo "Checking code formatting and quality..."
	${VENV}/bin/black --check .
	${VENV}/bin/flake8 .

# Prepare the data
prepare:
	@echo "Preprocessing data..."
	${VENV}/bin/python model_pipeline.py preprocess

# Train the model
train:
	@echo "Training the model..."
	${VENV}/bin/python model_pipeline.py train

# Run both prepare and train
all: prepare train

# Clean generated files
clean:
	@echo "Cleaning up generated files..."
	rm -f ${TRAINED_MODEL} ${SCALER} ${PREPROCESSED_DATA}

# Run tests (if applicable)
test:
	@echo "Running tests..."
	${VENV}/bin/pytest tests/

# Run the API and display Swagger UI URL
run-api:
	@echo "Starting API..."
	${VENV}/bin/python app.py &
	sleep 2
	@echo "API started. Open Swagger UI at: http://127.0.0.1:8000/docs"

.PHONY: install lint prepare train all clean test run-api
