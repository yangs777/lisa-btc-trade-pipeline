"""Vertex AI pipeline orchestration."""

from typing import Dict, Any, Optional
import json


class VertexPipelineOrchestrator:
    """Orchestrate pipelines on Vertex AI."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """Initialize Vertex orchestrator.
        
        Args:
            project_id: GCP project ID
            location: GCP location
        """
        self.project_id = project_id
        self.location = location
    
    def create_training_pipeline(
        self,
        dataset_uri: str,
        model_uri: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create training pipeline.
        
        Args:
            dataset_uri: GCS URI for dataset
            model_uri: GCS URI for model output
            config: Training configuration
            
        Returns:
            Pipeline specification
        """
        pipeline_spec = {
            "displayName": "Bitcoin Trading Model Training",
            "pipelineSpec": {
                "components": {
                    "data-preprocessing": {
                        "executorLabel": "exec-preprocess",
                        "inputDefinitions": {
                            "parameters": {
                                "dataset_uri": {"type": "STRING"}
                            }
                        },
                        "outputDefinitions": {
                            "parameters": {
                                "processed_uri": {"type": "STRING"}
                            }
                        }
                    },
                    "model-training": {
                        "executorLabel": "exec-train",
                        "inputDefinitions": {
                            "parameters": {
                                "data_uri": {"type": "STRING"},
                                "config": {"type": "STRING"}
                            }
                        },
                        "outputDefinitions": {
                            "parameters": {
                                "model_uri": {"type": "STRING"}
                            }
                        }
                    },
                    "model-evaluation": {
                        "executorLabel": "exec-evaluate",
                        "inputDefinitions": {
                            "parameters": {
                                "model_uri": {"type": "STRING"},
                                "test_data_uri": {"type": "STRING"}
                            }
                        },
                        "outputDefinitions": {
                            "parameters": {
                                "metrics": {"type": "STRING"}
                            }
                        }
                    }
                },
                "deploymentSpec": {
                    "executors": {
                        "exec-preprocess": {
                            "container": {
                                "image": f"gcr.io/{self.project_id}/btc-preprocess:latest",
                                "command": ["python", "-m", "src.data_processing.daily_preprocessor"],
                                "args": ["--input", "{{$.inputs.parameters['dataset_uri']}}"]
                            }
                        },
                        "exec-train": {
                            "container": {
                                "image": f"gcr.io/{self.project_id}/btc-train:latest",
                                "command": ["python", "-m", "src.main", "train"],
                                "args": [
                                    "--data", "{{$.inputs.parameters['data_uri']}}",
                                    "--config", "{{$.inputs.parameters['config']}}"
                                ]
                            }
                        },
                        "exec-evaluate": {
                            "container": {
                                "image": f"gcr.io/{self.project_id}/btc-evaluate:latest",
                                "command": ["python", "-m", "src.backtesting.engine"],
                                "args": [
                                    "--model", "{{$.inputs.parameters['model_uri']}}",
                                    "--data", "{{$.inputs.parameters['test_data_uri']}}"
                                ]
                            }
                        }
                    }
                },
                "pipelineInfo": {
                    "name": "btc-trading-pipeline"
                },
                "root": {
                    "dag": {
                        "tasks": {
                            "preprocess": {
                                "componentRef": {
                                    "name": "data-preprocessing"
                                },
                                "inputs": {
                                    "parameters": {
                                        "dataset_uri": {
                                            "runtimeValue": {
                                                "constantValue": {
                                                    "stringValue": dataset_uri
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "train": {
                                "componentRef": {
                                    "name": "model-training"
                                },
                                "dependentTasks": ["preprocess"],
                                "inputs": {
                                    "parameters": {
                                        "data_uri": {
                                            "taskOutputParameter": {
                                                "producerTask": "preprocess",
                                                "outputParameterKey": "processed_uri"
                                            }
                                        },
                                        "config": {
                                            "runtimeValue": {
                                                "constantValue": {
                                                    "stringValue": json.dumps(config or {})
                                                }
                                            }
                                        }
                                    }
                                }
                            },
                            "evaluate": {
                                "componentRef": {
                                    "name": "model-evaluation"
                                },
                                "dependentTasks": ["train"],
                                "inputs": {
                                    "parameters": {
                                        "model_uri": {
                                            "taskOutputParameter": {
                                                "producerTask": "train",
                                                "outputParameterKey": "model_uri"
                                            }
                                        },
                                        "test_data_uri": {
                                            "taskOutputParameter": {
                                                "producerTask": "preprocess",
                                                "outputParameterKey": "processed_uri"
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        return pipeline_spec
    
    def create_batch_prediction_pipeline(
        self,
        model_uri: str,
        input_uri: str,
        output_uri: str
    ) -> Dict[str, Any]:
        """Create batch prediction pipeline.
        
        Args:
            model_uri: GCS URI for model
            input_uri: GCS URI for input data
            output_uri: GCS URI for output predictions
            
        Returns:
            Pipeline specification
        """
        return {
            "displayName": "Bitcoin Trading Batch Prediction",
            "pipelineSpec": {
                "components": {
                    "batch-predict": {
                        "executorLabel": "exec-predict",
                        "inputDefinitions": {
                            "parameters": {
                                "model_uri": {"type": "STRING"},
                                "input_uri": {"type": "STRING"},
                                "output_uri": {"type": "STRING"}
                            }
                        }
                    }
                },
                "deploymentSpec": {
                    "executors": {
                        "exec-predict": {
                            "container": {
                                "image": f"gcr.io/{self.project_id}/btc-predict:latest",
                                "command": ["python", "-m", "src.pipeline.batch_predict"],
                                "args": [
                                    "--model", "{{$.inputs.parameters['model_uri']}}",
                                    "--input", "{{$.inputs.parameters['input_uri']}}",
                                    "--output", "{{$.inputs.parameters['output_uri']}}"
                                ]
                            }
                        }
                    }
                }
            }
        }