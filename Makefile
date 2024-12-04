ENV_NAME=flask_env
APP_FILE=app.py
REQUIREMENTS=requirements.txt
UPLOADS_DIR=static/uploads

.PHONY: all create_env install run clean

all: run

create_env:
	@echo "Creating virtual environment..."
	python3 -m venv $(ENV_NAME)

install: create_env
	@echo "Installing dependencies..."
	$(ENV_NAME)/bin/pip install --upgrade pip
	$(ENV_NAME)/bin/pip install -r $(REQUIREMENTS)

run:
	@echo "Starting Flask app..."
	$(ENV_NAME)/bin/python $(APP_FILE)

clean:
	@echo "Cleaning up..."
	rm -rf $(ENV_NAME)
	rm -rf __pycache__
	rm -rf $(UPLOADS_DIR)/*
	@echo "Cleanup complete."

