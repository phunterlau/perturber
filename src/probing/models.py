from __future__ import annotations

import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from .contracts import ModelRequest
from .errors import ModelPolicyError


def configure_huggingface_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    os.environ["HF_XET_CACHE"] = str(cache_dir / "xet")
    os.environ["HF_HUB_DISABLE_XET"] = "1"


class ModelManager:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir.resolve()
        configure_huggingface_cache(self.cache_dir)
        self._normalize_symbolic_references()

    def _normalize_symbolic_references(self) -> None:
        """Repair references written by probe versions that appended a newline.

        huggingface_hub 1.26 reads a symbolic ref verbatim.  A trailing newline
        therefore turns an otherwise complete offline snapshot into a cache
        miss.  Only rewrite refs whose trimmed value is a local snapshot hash;
        this keeps the migration bounded to valid Hugging Face cache entries.
        """
        for refs in self.cache_dir.glob("models--*/refs"):
            if not refs.is_dir():
                continue
            snapshots = refs.parent / "snapshots"
            for reference in refs.rglob("*"):
                if not reference.is_file():
                    continue
                try:
                    raw_revision = reference.read_text(encoding="utf-8")
                except OSError:
                    continue
                revision = raw_revision.strip()
                if (
                    raw_revision == revision
                    or not re.fullmatch(r"[0-9a-fA-F]{7,64}", revision)
                    or not (snapshots / revision).is_dir()
                ):
                    continue
                temporary = reference.with_suffix(reference.suffix + ".tmp")
                temporary.write_text(revision, encoding="utf-8")
                os.replace(temporary, reference)

    def _repository_directory(self, model_id: str) -> Path:
        return self.cache_dir / "models--" / model_id.replace("/", "--")

    def cached_snapshots(self, model_id: str) -> tuple[Path, ...]:
        directories = [
            self.cache_dir / f"models--{model_id.replace('/', '--')}" / "snapshots",
            self.cache_dir / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots",
        ]
        snapshots: list[Path] = []
        for directory in directories:
            if directory.is_dir():
                snapshots.extend(path for path in directory.iterdir() if path.is_dir())
        return tuple(sorted(set(snapshots)))

    def is_cached(self, request: ModelRequest) -> bool:
        snapshots = self.cached_snapshots(request.id)
        if not snapshots:
            return False
        if request.revision is None:
            return True
        return any(path.name.startswith(request.revision) for path in snapshots)

    def resolve_cached_snapshot(self, request: ModelRequest) -> Path:
        """Return the immutable local directory that satisfies a model request."""
        snapshots = self.cached_snapshots(request.id)
        if request.revision is not None:
            matches = tuple(
                path for path in snapshots if path.name.startswith(request.revision)
            )
        else:
            reference = self._repository_directory(request.id) / "refs" / "main"
            try:
                revision = reference.read_text(encoding="utf-8").strip()
            except OSError:
                revision = ""
            matches = tuple(path for path in snapshots if path.name == revision)
            if not matches and len(snapshots) == 1:
                matches = snapshots
        if len(matches) != 1:
            raise ModelPolicyError(
                f"cannot resolve one cached snapshot for model {request.id!r}",
                hint="Pin model.revision to one of the cached snapshot hashes.",
                details={
                    "model_id": request.id,
                    "requested_revision": request.revision,
                    "snapshots": [path.name for path in snapshots],
                },
            )
        return matches[0]

    def inspect_cached(self, request: ModelRequest) -> dict[str, Any]:
        snapshots = self.cached_snapshots(request.id)
        model_types: set[str] = set()
        for snapshot in snapshots:
            config_path = snapshot / "config.json"
            try:
                value = json.loads(config_path.read_text(encoding="utf-8"))
                model_type = value.get("model_type")
                if isinstance(model_type, str):
                    model_types.add(model_type)
            except (OSError, ValueError):
                continue
        return {
            "model_id": request.id,
            "requested_revision": request.revision,
            "cached": self.is_cached(request),
            "snapshots": [path.name for path in snapshots],
            "model_types": sorted(model_types),
            "cache_dir": str(self.cache_dir),
        }

    def remote_size(self, request: ModelRequest) -> tuple[int, str | None]:
        info = HfApi().model_info(
            request.id,
            revision=request.revision,
            files_metadata=True,
        )
        unknown = [sibling.rfilename for sibling in info.siblings if sibling.size is None]
        if unknown:
            raise ModelPolicyError(
                "cannot enforce the download budget because the repository has "
                "files with unknown sizes",
                details={
                    "model_id": request.id,
                    "unknown_size_files": unknown,
                },
            )
        sizes = [sibling.size for sibling in info.siblings]
        return sum(sizes), info.sha

    def fetch(self, request: ModelRequest, *, max_download_bytes: int) -> Path:
        size, resolved_revision = self.remote_size(request)
        if not resolved_revision:
            raise ModelPolicyError(
                "the model registry did not return an immutable revision",
                details={"model_id": request.id},
            )
        if size > max_download_bytes:
            raise ModelPolicyError(
                f"model snapshot requires {size} bytes, exceeding the download budget "
                f"of {max_download_bytes}",
                details={
                    "model_id": request.id,
                    "resolved_revision": resolved_revision,
                    "size_bytes": size,
                    "max_download_bytes": max_download_bytes,
                },
            )
        downloaded = snapshot_download(
            repo_id=request.id,
            # Bind acquisition to the exact revision whose file sizes were
            # checked. Otherwise a moving branch could change between budget
            # inspection and download.
            revision=resolved_revision,
            cache_dir=str(self.cache_dir),
            local_files_only=False,
        )
        path = Path(downloaded)
        self._record_reference(request, resolved_revision, path)
        return path

    @staticmethod
    def _record_reference(
        request: ModelRequest, resolved_revision: str, snapshot: Path
    ) -> None:
        """Keep symbolic Hugging Face revisions resolvable after a pinned fetch."""
        reference = request.revision or "main"
        if re.fullmatch(r"[0-9a-fA-F]{7,64}", reference):
            return
        candidate = PurePosixPath(reference)
        if (
            candidate.is_absolute()
            or "\\" in reference
            or str(candidate) != reference
            or any(part in {"", ".", ".."} for part in candidate.parts)
        ):
            raise ModelPolicyError(
                f"revision {reference!r} cannot be represented safely in the local cache"
            )
        if snapshot.parent.name != "snapshots":
            raise ModelPolicyError(
                "downloaded snapshot path does not match the Hugging Face cache layout",
                details={"snapshot": str(snapshot)},
            )
        refs = snapshot.parent.parent / "refs"
        ref_path = refs.joinpath(*candidate.parts)
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = ref_path.with_suffix(ref_path.suffix + ".tmp")
        # huggingface_hub resolves this value verbatim; do not append a newline.
        temporary.write_text(resolved_revision, encoding="utf-8")
        os.replace(temporary, ref_path)

    def ensure_available(
        self,
        request: ModelRequest,
        *,
        allow_download: bool,
        max_download_bytes: int | None,
    ) -> None:
        if self.is_cached(request):
            return
        if not allow_download:
            raise ModelPolicyError(
                f"model {request.id!r} is not cached and downloads are disabled",
                hint="Run 'probe model fetch' with an explicit byte budget first.",
                details={"model_id": request.id, "cache_dir": str(self.cache_dir)},
            )
        if max_download_bytes is None:
            raise ModelPolicyError("a download byte budget is required")
        self.fetch(request, max_download_bytes=max_download_bytes)
