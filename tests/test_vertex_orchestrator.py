"""Test coverage for Vertex AI pipeline orchestrator."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json

from src.pipeline.vertex_orchestrator import VertexPipelineOrchestrator


class TestVertexPipelineOrchestrator:
    """Test VertexPipelineOrchestrator class."""
    
    def test_initialization(self) -> None:
        """Test orchestrator initialization."""
        orchestrator = VertexPipelineOrchestrator(
            project_id="test-project",
            location="us-central1"
        )
        
        assert orchestrator.project_id == "test-project"
        assert orchestrator.location == "us-central1"
    
    def test_initialization_with_custom_location(self) -> None:
        """Test orchestrator initialization with custom location."""
        orchestrator = VertexPipelineOrchestrator(
            project_id="test-project",
            location="us-west1"
        )
        
        assert orchestrator.project_id == "test-project"
        assert orchestrator.location == "us-west1"
    
    def test_create_training_pipeline(self) -> None:
        """Test creating training pipeline."""
        orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
        
        config = {
            "machine_type": "n1-standard-8",
            "accelerator_type": "NVIDIA_TESLA_V100",
            "accelerator_count": 1
        }
        
        result = orchestrator.create_training_pipeline(
            dataset_uri="gs://test-data/dataset",
            model_uri="gs://test-models/model",
            config=config
        )
        
        # Verify result structure
        assert "displayName" in result
        assert result["displayName"] == "Bitcoin Trading Model Training"
        assert "pipelineSpec" in result
        assert "components" in result["pipelineSpec"]
        
        # Verify components
        components = result["pipelineSpec"]["components"]
        assert "data-preprocessing" in components
        assert "model-training" in components
        assert "model-evaluation" in components
        
        # Verify root spec
        assert "root" in result["pipelineSpec"]
        root = result["pipelineSpec"]["root"]
        assert "dag" in root
        assert "tasks" in root["dag"]
    
    def test_create_training_pipeline_no_config(self) -> None:
        """Test creating training pipeline without config."""
        orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
        
        result = orchestrator.create_training_pipeline(
            dataset_uri="gs://test-data/dataset",
            model_uri="gs://test-models/model"
        )
        
        # Verify result
        assert "displayName" in result
        assert "pipelineSpec" in result
    
    def test_create_batch_prediction_pipeline(self) -> None:
        """Test creating batch prediction pipeline."""
        orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
        
        result = orchestrator.create_batch_prediction_pipeline(
            model_uri="gs://test-models/model",
            input_uri="gs://test-data/input.jsonl",
            output_uri="gs://test-output/"
        )
        
        # Verify result structure
        assert "displayName" in result
        assert result["displayName"] == "Bitcoin Trading Batch Prediction"
        assert "pipelineSpec" in result
        assert "components" in result["pipelineSpec"]
        
        # Verify components
        components = result["pipelineSpec"]["components"]
        assert "batch-predict" in components
        
        # Verify deployment spec
        assert "deploymentSpec" in result["pipelineSpec"]
        assert "executors" in result["pipelineSpec"]["deploymentSpec"]
        assert "exec-predict" in result["pipelineSpec"]["deploymentSpec"]["executors"]
    


class TestPipelineComponents:
    """Test individual pipeline components."""
    
    def test_data_preprocessing_component(self) -> None:
        """Test data preprocessing component spec."""
        orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
        pipeline = orchestrator.create_training_pipeline(
            dataset_uri="gs://test/data",
            model_uri="gs://test/model"
        )
        
        component = pipeline["pipelineSpec"]["components"]["data-preprocessing"]
        
        # Check input definitions
        assert "inputDefinitions" in component
        assert "parameters" in component["inputDefinitions"]
        assert "dataset_uri" in component["inputDefinitions"]["parameters"]
        
        # Check output definitions  
        assert "outputDefinitions" in component
        assert "parameters" in component["outputDefinitions"]
        assert "processed_uri" in component["outputDefinitions"]["parameters"]
        
        # Check executor
        assert "executorLabel" in component
        assert component["executorLabel"] == "exec-preprocess"
    
    def test_model_training_component(self) -> None:
        """Test model training component spec."""
        orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
        pipeline = orchestrator.create_training_pipeline(
            dataset_uri="gs://test/data",
            model_uri="gs://test/model",
            config={"epochs": 100}
        )
        
        component = pipeline["pipelineSpec"]["components"]["model-training"]
        
        # Check input definitions
        assert "inputDefinitions" in component
        inputs = component["inputDefinitions"]["parameters"]
        assert "data_uri" in inputs
        assert "config" in inputs
        
        # Check output definitions
        assert "outputDefinitions" in component
        outputs = component["outputDefinitions"]["parameters"]
        assert "model_uri" in outputs


def test_module_import() -> None:
    """Test that the module can be imported."""
    from src.pipeline import vertex_orchestrator
    assert hasattr(vertex_orchestrator, 'VertexPipelineOrchestrator')