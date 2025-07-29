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
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            mock_aiplatform.init = Mock()
            
            orchestrator = VertexPipelineOrchestrator(
                project_id="test-project",
                location="us-central1",
                staging_bucket="gs://test-bucket"
            )
            
            assert orchestrator.project_id == "test-project"
            assert orchestrator.location == "us-central1"
            assert orchestrator.staging_bucket == "gs://test-bucket"
            
            mock_aiplatform.init.assert_called_once_with(
                project="test-project",
                location="us-central1",
                staging_bucket="gs://test-bucket"
            )
    
    def test_create_training_pipeline(self) -> None:
        """Test creating training pipeline."""
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            # Setup mocks
            mock_job = Mock()
            mock_job.run = Mock()
            mock_aiplatform.CustomTrainingJob.return_value = mock_job
            
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
            
            # Verify job creation
            mock_aiplatform.CustomTrainingJob.assert_called_once()
            call_args = mock_aiplatform.CustomTrainingJob.call_args[1]
            assert call_args["display_name"] == "btc-trading-training"
            assert "gcr.io" in call_args["container_uri"]
            
            # Verify job run
            mock_job.run.assert_called_once()
            run_args = mock_job.run.call_args[1]
            assert run_args["machine_type"] == "n1-standard-8"
            assert run_args["accelerator_type"] == "NVIDIA_TESLA_V100"
            
            # Verify result
            assert result["status"] == "submitted"
            assert result["job_name"] == mock_job.resource_name
    
    def test_create_batch_prediction_pipeline(self) -> None:
        """Test creating batch prediction pipeline."""
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            # Setup mocks
            mock_job = Mock()
            mock_job.run = Mock()
            mock_job.resource_name = "projects/test/locations/us/batchPredictionJobs/123"
            mock_aiplatform.BatchPredictionJob.return_value = mock_job
            
            mock_model = Mock()
            mock_model.resource_name = "projects/test/locations/us/models/456"
            mock_aiplatform.Model.return_value = mock_model
            
            orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
            
            result = orchestrator.create_batch_prediction_pipeline(
                model_id="model-123",
                input_uri="gs://test-data/input.jsonl",
                output_uri="gs://test-output/"
            )
            
            # Verify model loading
            mock_aiplatform.Model.assert_called_once_with(model_name="model-123")
            
            # Verify job creation
            mock_aiplatform.BatchPredictionJob.assert_called_once()
            call_args = mock_aiplatform.BatchPredictionJob.call_args[1]
            assert call_args["display_name"] == "btc-trading-batch-prediction"
            assert call_args["model_name"] == mock_model.resource_name
            assert call_args["instances_format"] == "jsonl"
            assert call_args["predictions_format"] == "jsonl"
            
            # Verify job run
            mock_job.run.assert_called_once()
            
            # Verify result
            assert result["status"] == "submitted"
            assert result["job_name"] == "projects/test/locations/us/batchPredictionJobs/123"
    
    def test_create_pipeline_with_components(self) -> None:
        """Test creating full pipeline with components."""
        with patch('src.pipeline.vertex_orchestrator.compiler') as mock_compiler:
            with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
                # Setup mocks
                mock_job = Mock()
                mock_job.submit = Mock()
                mock_job.resource_name = "projects/test/pipelineJobs/789"
                mock_aiplatform.PipelineJob.return_value = mock_job
                
                orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
                
                # Create pipeline
                @orchestrator.component(
                    base_image="python:3.9",
                    packages=["pandas", "numpy"]
                )
                def preprocess_data(input_path: str, output_path: str) -> None:
                    """Preprocess data component."""
                    pass
                
                @orchestrator.component()
                def train_model(data_path: str, model_path: str) -> None:
                    """Train model component."""
                    pass
                
                @orchestrator.pipeline(name="btc-trading-pipeline")
                def trading_pipeline(input_data: str) -> None:
                    """Full trading pipeline."""
                    preprocess_task = preprocess_data(
                        input_path=input_data,
                        output_path="gs://temp/processed"
                    )
                    train_task = train_model(
                        data_path=preprocess_task.output,
                        model_path="gs://models/output"
                    )
                
                # Run pipeline
                result = orchestrator.run_pipeline(
                    pipeline_func=trading_pipeline,
                    parameter_values={"input_data": "gs://data/input"}
                )
                
                # Verify compilation
                mock_compiler.Compiler.assert_called_once()
                
                # Verify job creation
                mock_aiplatform.PipelineJob.assert_called_once()
                
                # Verify job submission
                mock_job.submit.assert_called_once()
                
                # Verify result
                assert result["status"] == "submitted"
                assert result["job_name"] == "projects/test/pipelineJobs/789"
    
    def test_monitor_job(self) -> None:
        """Test job monitoring."""
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            # Setup mock job states
            mock_job = Mock()
            mock_job.state = "JOB_STATE_PENDING"
            mock_job.error = None
            
            def get_job(name):
                return mock_job
            
            mock_aiplatform.get_job = get_job
            
            orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
            
            # Test pending state
            status = orchestrator.monitor_job("projects/test/jobs/123", job_type="training")
            assert status["state"] == "JOB_STATE_PENDING"
            assert status["is_complete"] == False
            assert status["is_failed"] == False
            
            # Test running state
            mock_job.state = "JOB_STATE_RUNNING"
            status = orchestrator.monitor_job("projects/test/jobs/123", job_type="training")
            assert status["state"] == "JOB_STATE_RUNNING"
            assert status["is_complete"] == False
            
            # Test succeeded state
            mock_job.state = "JOB_STATE_SUCCEEDED"
            status = orchestrator.monitor_job("projects/test/jobs/123", job_type="training")
            assert status["state"] == "JOB_STATE_SUCCEEDED"
            assert status["is_complete"] == True
            assert status["is_failed"] == False
            
            # Test failed state
            mock_job.state = "JOB_STATE_FAILED"
            mock_job.error = Mock(message="Training failed")
            status = orchestrator.monitor_job("projects/test/jobs/123", job_type="training")
            assert status["state"] == "JOB_STATE_FAILED"
            assert status["is_complete"] == True
            assert status["is_failed"] == True
            assert "Training failed" in status["error"]
    
    def test_list_models(self) -> None:
        """Test listing models."""
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            # Setup mock models
            mock_models = [
                Mock(resource_name="model1", display_name="Model 1", 
                     create_time="2024-01-01", update_time="2024-01-02"),
                Mock(resource_name="model2", display_name="Model 2",
                     create_time="2024-01-03", update_time="2024-01-04")
            ]
            mock_aiplatform.Model.list.return_value = mock_models
            
            orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
            
            models = orchestrator.list_models(filter_str="btc-trading")
            
            # Verify API call
            mock_aiplatform.Model.list.assert_called_once_with(filter="btc-trading")
            
            # Verify result
            assert len(models) == 2
            assert models[0]["resource_name"] == "model1"
            assert models[0]["display_name"] == "Model 1"
            assert models[1]["resource_name"] == "model2"
    
    def test_deploy_model(self) -> None:
        """Test model deployment."""
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            # Setup mocks
            mock_model = Mock()
            mock_endpoint = Mock()
            mock_endpoint.resource_name = "projects/test/endpoints/123"
            mock_model.deploy = Mock(return_value=mock_endpoint)
            mock_aiplatform.Model.return_value = mock_model
            
            orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
            
            endpoint = orchestrator.deploy_model(
                model_id="model-123",
                endpoint_name="btc-trading-endpoint",
                machine_type="n1-standard-4",
                min_replicas=1,
                max_replicas=3
            )
            
            # Verify model loading
            mock_aiplatform.Model.assert_called_once_with(model_name="model-123")
            
            # Verify deployment
            mock_model.deploy.assert_called_once()
            deploy_args = mock_model.deploy.call_args[1]
            assert deploy_args["deployed_model_display_name"] == "btc-trading-endpoint"
            assert deploy_args["machine_type"] == "n1-standard-4"
            assert deploy_args["min_replica_count"] == 1
            assert deploy_args["max_replica_count"] == 3
            
            # Verify result
            assert endpoint["endpoint_name"] == "projects/test/endpoints/123"
            assert endpoint["model_id"] == "model-123"
    
    def test_error_handling(self) -> None:
        """Test error handling in various operations."""
        with patch('src.pipeline.vertex_orchestrator.aiplatform') as mock_aiplatform:
            # Setup to raise exception
            mock_aiplatform.init.side_effect = Exception("Authentication failed")
            
            with pytest.raises(Exception, match="Authentication failed"):
                orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
            
            # Reset and test job creation error
            mock_aiplatform.init.side_effect = None
            orchestrator = VertexPipelineOrchestrator("test-project", "us-central1")
            
            mock_aiplatform.CustomTrainingJob.side_effect = Exception("Invalid config")
            
            result = orchestrator.create_training_pipeline(
                dataset_uri="gs://data",
                model_uri="gs://model"
            )
            
            assert result["status"] == "failed"
            assert "Invalid config" in result["error"]