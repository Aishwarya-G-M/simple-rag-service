import logging
import json
from datetime import datetime

logger = logging.getLogger("simple_rag_metrics")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # We’ll attach metrics as a plain dict via extra={"metrics": ...}
        metrics = getattr(record, "metrics", None)
        if metrics is not None:
            payload["metrics"] = metrics

        return json.dumps(payload)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)