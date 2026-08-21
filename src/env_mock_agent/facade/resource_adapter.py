from __future__ import annotations

from env_mock_agent.facade.resource_v2 import (
    AttachmentResourceGrantV2,
    AttachmentResourceUsageV2,
    ResourceBoundPublicSourceFetchResultV2,
    ResourceBoundPublicSourceSearchResultV2,
    attachment_resource_grant_ref,
    validate_attachment_resource_grant_identity,
)
from env_mock_agent.facade.retrieval_v2 import (
    PublicSourceFetchRequestV2,
    PublicSourceRetrievalFacade,
    PublicSourceSearchRequestV2,
    public_source_fetch_request_ref,
    public_source_search_request_ref,
)


class RegistryResourceBoundRetrievalFacade:
    def __init__(self, retrieval: PublicSourceRetrievalFacade) -> None:
        self._retrieval = retrieval

    async def search_with_resources(
        self,
        request: PublicSourceSearchRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundPublicSourceSearchResultV2:
        validate_attachment_resource_grant_identity(grant)
        if grant.operation_ref != public_source_search_request_ref(request):
            raise ValueError("resource grant does not bind the search request")
        usage = AttachmentResourceUsageV2(
            process_starts=0,
            renderer_operations=0,
            network_requests=1,
            retained_storage_bytes=0,
        )
        grant.validate_usage(usage)
        result = await self._retrieval.search(request)
        return ResourceBoundPublicSourceSearchResultV2(
            resource_grant_ref=attachment_resource_grant_ref(grant),
            search_result=result,
            usage=usage,
        )

    async def fetch_with_resources(
        self,
        request: PublicSourceFetchRequestV2,
        grant: AttachmentResourceGrantV2,
    ) -> ResourceBoundPublicSourceFetchResultV2:
        validate_attachment_resource_grant_identity(grant)
        if grant.operation_ref != public_source_fetch_request_ref(request):
            raise ValueError("resource grant does not bind the fetch request")
        usage = AttachmentResourceUsageV2(
            process_starts=0,
            renderer_operations=0,
            network_requests=1,
            retained_storage_bytes=0,
        )
        grant.validate_usage(usage)
        result = await self._retrieval.fetch(request)
        return ResourceBoundPublicSourceFetchResultV2(
            resource_grant_ref=attachment_resource_grant_ref(grant),
            fetch_result=result,
            usage=usage,
        )
