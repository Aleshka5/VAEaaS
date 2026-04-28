class MLFlowRepository:
    """
    Класс для сохранения KVAE модели в mlflow и загрузки её из mlflow.
    Важно, что модель состоит из енкодера и декодера. Это должны быть 2 разные модели.
    """

    def __init__(self, tracking_uri: str | None = None, registry_uri: str | None = None):
        import mlflow
        import os
        from src.config import get_settings

        self.mlflow = mlflow
        settings = get_settings()
        resolved_tracking_uri = tracking_uri or settings.mlflow_tracking_uri
        resolved_registry_uri = registry_uri or settings.mlflow_registry_uri

        if not resolved_tracking_uri:
            raise ValueError(
                "MLFLOW_TRACKING_URI не задан. "
                "Укажите его в .env или передайте --mlflow-tracking-uri."
            )
        if not resolved_registry_uri:
            raise ValueError(
                "MLFLOW_REGISTRY_URI не задан. "
                "Укажите его в .env или передайте --mlflow-registry-uri."
            )

        env_overrides = {
            "AWS_ACCESS_KEY_ID": settings.aws_access_key_id,
            "AWS_SECRET_ACCESS_KEY": settings.aws_secret_access_key,
            "AWS_DEFAULT_REGION": settings.aws_default_region,
            "S3_ENDPOINT_URL": settings.s3_endpoint_url,
            "MLFLOW_S3_ENDPOINT_URL": settings.mlflow_s3_endpoint_url,
        }
        for key, value in env_overrides.items():
            if value and not os.getenv(key):
                os.environ[key] = value

        if resolved_tracking_uri:
            mlflow.set_tracking_uri(resolved_tracking_uri)
        if resolved_registry_uri:
            mlflow.set_registry_uri(resolved_registry_uri)
        active_tracking_uri = mlflow.get_tracking_uri()
        if str(active_tracking_uri).startswith("file:"):
            raise ValueError(
                f"Обнаружен локальный file-based tracking URI: '{active_tracking_uri}'. "
                "Ожидается URI запущенного MLflow server (например, http://host:5000)."
            )

        self.tracking_uri = resolved_tracking_uri
        self.registry_uri = resolved_registry_uri
        self.settings = settings
        self.experiment_name = settings.mlflow_experiment_name
        self.client = mlflow.tracking.MlflowClient()

    def _get_experiment_id(self) -> str:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            raise ValueError(
                f"Эксперимент '{self.experiment_name}' не найден в MLflow. "
                "Создайте его заранее или укажите корректный MLFLOW_EXPERIMENT_NAME."
            )
        return experiment.experiment_id

    def _latest_model_version(self, model_name: str) -> str:
        versions = self.client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise ValueError(f"В registry не найдена модель с именем '{model_name}'.")
        latest = max(versions, key=lambda v: int(v.version))
        return str(latest.version)

    @staticmethod
    def _resolve_model_uri(model_name: str, stage: str | None = None, version: str | None = None) -> str:
        if version:
            return f"models:/{model_name}/{version}"
        if stage:
            return f"models:/{model_name}/{stage}"
        return f"models:/{model_name}/latest"

    def save_encoder_decoder(
        self,
        encoder,
        decoder,
        *,
        encoder_model_name: str,
        decoder_model_name: str,
        run_name: str = "kvae-components",
    ) -> dict[str, str]:
        experiment_id = self._get_experiment_id()
        with self.mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            self.mlflow.pytorch.log_model(
                pytorch_model=encoder,
                name="encoder",
                registered_model_name=encoder_model_name,
            )
            self.mlflow.pytorch.log_model(
                pytorch_model=decoder,
                name="decoder",
                registered_model_name=decoder_model_name,
            )
            return {"run_id": run.info.run_id}

    def save_kvae_from_pretrained(
        self,
        *,
        pretrained_model_path: str,
        encoder_model_name: str,
        decoder_model_name: str,
        run_name: str = "kvae-components",
        subfolder: str = "diffusers",
        local_files_only: bool = True,
        torch_dtype=None,
    ) -> dict[str, str]:
        from src.utils.decoder import KVAEDecoder
        from src.utils.encoder import KVAEEncoder

        encoder = KVAEEncoder.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        decoder = KVAEDecoder.from_pretrained(
            pretrained_model_path=pretrained_model_path,
            subfolder=subfolder,
            local_files_only=local_files_only,
            torch_dtype=torch_dtype,
        )
        return self.save_encoder_decoder(
            encoder,
            decoder,
            encoder_model_name=encoder_model_name,
            decoder_model_name=decoder_model_name,
            run_name=run_name,
        )

    def log_artifacts(
        self,
        artifact_paths: list[str],
        *,
        run_name: str,
        artifact_subdir: str = "artifacts",
        tags: dict[str, str] | None = None,
    ) -> dict[str, str]:
        from pathlib import Path

        if not artifact_paths:
            raise ValueError("Список artifact_paths пуст.")

        experiment_id = self._get_experiment_id()
        with self.mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
            if tags:
                self.mlflow.set_tags(tags)

            for artifact_path in artifact_paths:
                local_path = Path(artifact_path)
                if not local_path.exists():
                    raise FileNotFoundError(f"Artifact file not found: {artifact_path}")
                self.mlflow.log_artifact(local_path=str(local_path), artifact_path=artifact_subdir)

            return {"run_id": run.info.run_id}

    def _load_encoder_from_local(self, device: str):
        from src.utils.encoder import KVAEEncoder

        if not self.settings.kvae_pretrained_path:
            raise ValueError(
                "Локальный fallback для encoder недоступен: не задан KVAE_PRETRAINED_PATH в .env."
            )
        return KVAEEncoder.from_pretrained(
            pretrained_model_path=str(self.settings.kvae_pretrained_path),
            subfolder=self.settings.kvae_subfolder,
            local_files_only=True,
        ).to(device)

    def _load_decoder_from_local(self, device: str):
        from src.utils.decoder import KVAEDecoder

        if not self.settings.kvae_pretrained_path:
            raise ValueError(
                "Локальный fallback для decoder недоступен: не задан KVAE_PRETRAINED_PATH в .env."
            )
        return KVAEDecoder.from_pretrained(
            pretrained_model_path=str(self.settings.kvae_pretrained_path),
            subfolder=self.settings.kvae_subfolder,
            local_files_only=True,
        ).to(device)

    def load_encoder(
        self,
        model_name: str,
        *,
        stage: str | None = None,
        version: str | None = None,
        device: str = "cpu",
    ):
        import warnings

        try:
            model_uri = self._resolve_model_uri(model_name=model_name, stage=stage, version=version)
            if model_uri.endswith("/latest"):
                resolved_version = self._latest_model_version(model_name=model_name)
                model_uri = f"models:/{model_name}/{resolved_version}"
            return self.mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
        except Exception as error:  # noqa: BLE001
            warnings.warn(
                f"Не удалось загрузить encoder из MLflow ({error}). "
                "Пробуем локальный fallback из KVAE_PRETRAINED_PATH.",
                stacklevel=2,
            )
            return self._load_encoder_from_local(device=device)

    def load_decoder(
        self,
        model_name: str,
        *,
        stage: str | None = None,
        version: str | None = None,
        device: str = "cpu",
    ):
        import warnings

        try:
            model_uri = self._resolve_model_uri(model_name=model_name, stage=stage, version=version)
            if model_uri.endswith("/latest"):
                resolved_version = self._latest_model_version(model_name=model_name)
                model_uri = f"models:/{model_name}/{resolved_version}"
            return self.mlflow.pytorch.load_model(model_uri=model_uri, map_location=device)
        except Exception as error:  # noqa: BLE001
            warnings.warn(
                f"Не удалось загрузить decoder из MLflow ({error}). "
                "Пробуем локальный fallback из KVAE_PRETRAINED_PATH.",
                stacklevel=2,
            )
            return self._load_decoder_from_local(device=device)
