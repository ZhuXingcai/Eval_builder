from env_mock_agent.profiles.base import PackageProfile
from env_mock_agent.profiles.cc import CcProfile
from env_mock_agent.profiles.generic import GenericProfile
from env_mock_agent.profiles.lh import LhProfile


def get_profile(name: str) -> PackageProfile:
    normalized = name.strip().lower()
    if normalized == "generic":
        return GenericProfile()
    if normalized == "cc":
        return CcProfile()
    if normalized == "lh":
        return LhProfile()
    raise KeyError(f"unknown profile: {name}")


__all__ = ["CcProfile", "GenericProfile", "LhProfile", "PackageProfile", "get_profile"]
