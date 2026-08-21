"""Transport-neutral AI Gateway for governed Eval Factory model work."""

from eval_factory.ai_gateway.invocation import (
    EmbeddedAIGateway,
    ProviderInvocationResult,
)
from eval_factory.ai_gateway.model_catalog import (
    ModelCatalog,
    ModelCatalogError,
    ModelRouteFacts,
)
from eval_factory.ai_gateway.policy import GatewayPolicyError, GatewaySecurityPolicy
from eval_factory.ai_gateway.prompt_registry import PromptRegistry
from eval_factory.ai_gateway.protocols import AIGateway, GatewayProvider
from eval_factory.ai_gateway.rag import (
    PurposeBoundRAGGateway,
    RAGAuthorizationError,
    RAGRetrievedDocument,
    RAGRetriever,
)
from eval_factory.ai_gateway.receipts import (
    GatewayRecordConflictError,
    GatewayRecordError,
    GatewayRecordIntegrityError,
    GatewayRecordStore,
)
from eval_factory.ai_gateway.routing import (
    ModelRouteBlockedError,
    ModelRouteError,
    ModelRouter,
)

__all__ = [
    "AIGateway",
    "EmbeddedAIGateway",
    "GatewayPolicyError",
    "GatewayProvider",
    "GatewayRecordConflictError",
    "GatewayRecordError",
    "GatewayRecordIntegrityError",
    "GatewayRecordStore",
    "GatewaySecurityPolicy",
    "ModelCatalog",
    "ModelCatalogError",
    "ModelRouteBlockedError",
    "ModelRouteError",
    "ModelRouteFacts",
    "ModelRouter",
    "PromptRegistry",
    "ProviderInvocationResult",
    "PurposeBoundRAGGateway",
    "RAGAuthorizationError",
    "RAGRetrievedDocument",
    "RAGRetriever",
]
