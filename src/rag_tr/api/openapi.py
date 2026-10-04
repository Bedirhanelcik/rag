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

Ikinci is: yazma uclarinin Bearer token istedigini belgede DUYURMAK. Yetki
kontrolu `api.auth` icinde basligi elle okudugu icin FastAPI bunu kendi
basina kesfedemiyordu; sonucta Swagger UI'da "Authorize" dugmesi hic
cikmiyor ve `/docs` uzerinden token gonderilemiyordu -- yani dagitilmis
arayuzden `/upload`, `/ingest`, silme ve sifirlama uclari denenemiyordu
(hepsi 401 donuyordu, token verebilecek bir alan yoktu).

ONEMLI: bu YALNIZCA dokumantasyon ciktisini etkiler. Uc noktalarin calisma
zamani davranisi, dogrulamasi ve yanitlari aynen kalir -- `/upload` ve
`/ingest` zaten gercek `UploadFile` aliyordu, yalnizca Swagger onlari yanlis
gosteriyordu; yetki kontrolu de zaten vardi, yalnizca belgede yazmiyordu.
"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

#: Pydantic'in ikili alanlar icin yazdigi isaret.
_BINARY_MEDIA_TYPE = "application/octet-stream"

#: Belgede tanimlanan guvenlik semasinin adi; Swagger UI "Authorize"
#: penceresinde bu adi gosterir.
_SECURITY_SCHEME = "BearerToken"

#: Token isteyen uc noktalar: (HTTP yontemi, yol). `api.auth.require_write_token`
#: cagiran her uc burada olmali. Tek kaynak olarak duruyor ki belge ile gercek
#: davranis ayri dusmesin; `tests/test_openapi_schema.py` ikisini karsilastirir.
PROTECTED_OPERATIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("post", "/ingest"),
        ("post", "/upload"),
        ("delete", "/documents/{source_file}"),
        ("post", "/documents/reset"),
    }
)


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


def _declare_write_token_scheme(schema: dict) -> None:
    """Yazma uclarini Bearer token isteyen islemler olarak isaretler.

    Sema YALNIZCA belgelenir; dogrulamayi yine `api.auth` yapar. Guvenlik
    islem bazinda ekleniyor, genel (global) olarak degil: okuma uclari token
    istemiyor ve Swagger'in onlara da baslik eklemesi yanlis bilgi verirdi.
    """
    components = schema.setdefault("components", {})
    components.setdefault("securitySchemes", {})[_SECURITY_SCHEME] = {
        "type": "http",
        "scheme": "bearer",
        "description": (
            "Korpusu değiştiren uç noktalar için `INGEST_API_TOKEN`. "
            "Sunucuda token tanımlı değilse bu uçlar 503 döner."
        ),
    }

    for path, operations in schema.get("paths", {}).items():
        for method, operation in operations.items():
            if (method.lower(), path) in PROTECTED_OPERATIONS:
                operation["security"] = [{_SECURITY_SCHEME: []}]


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
    _declare_write_token_scheme(schema)
    app.openapi_schema = schema
    return schema
