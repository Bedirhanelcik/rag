"""OpenAPI belgesinde dosya alanlarini Swagger UI'nin tanidigi bicime cevirir.

Sorun: FastAPI (0.141 / Pydantic v2) bir `UploadFile` alanini JSON Schema
2020-12 bicimiyle yaziyor:

    {"type": "string", "contentMediaType": "application/octet-stream"}

Bu OpenAPI 3.1 acisindan dogru, ama Swagger UI dosya secme alanini
`format: "binary"` gorunce ciziyor. `contentMediaType` ile karsilastiginda
alani sira dan bir metin dizisi (`array<string>`) sanip metin kutusu
gosteriyor -- yani arayuzden PDF yuklenemiyor.

`File(...)` eklemek ya da `openapi_version` dusurmek bu ciktiyi
degistirmiyor (ucu de olculdu), cunku sema Pydantic tarafindan uretiliyor.
Bu yuzden duzeltme belgenin kendisinde yapiliyor: uretilen sema bir kez
gezilip ikili (binary) alanlar `format: "binary"` olarak yeniden yaziliyor.

ONEMLI: bu YALNIZCA dokumantasyon ciktisini etkiler. Uc noktalarin calisma
zamani davranisi, dogrulamasi ve yanitlari aynen kalir -- `/upload` ve
`/ingest` zaten gercek `UploadFile` aliyordu, yalnizca Swagger onlari yanlis
gosteriyordu.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

#: Pydantic'in ikili alanlar icin yazdigi isaret.
_BINARY_MEDIA_TYPE = "application/octet-stream"


def _normalise_binary_fields(node: Any) -> None:
    """Semayi yerinde gezerek ikili alanlari `format: binary` yapar."""
    if isinstance(node, dict):
        if node.get("type") == "string" and node.get("contentMediaType") == _BINARY_MEDIA_TYPE:
            node.pop("contentMediaType")
            node["format"] = "binary"
        for value in node.values():
            _normalise_binary_fields(value)
    elif isinstance(node, list):
        for item in node:
            _normalise_binary_fields(item)


def build_openapi(app: FastAPI) -> dict:
    """`app.openapi()` yerine gecer. Sonuc bir kez uretilip onbellege alinir."""
    if app.openapi_schema is not None:
        return app.openapi_schema

    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    _normalise_binary_fields(schema)
    app.openapi_schema = schema
    return schema
