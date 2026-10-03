"""Tek tip hata govdesi.

Kendi uc noktalarimiz zaten `{"detail": {"code": ..., "message": ...}}`
donduruyordu, ama framework kaynakli hatalar farkli sekiller uretiyordu:
dogrulama hatalari `detail`'i bir liste olarak, 404/405 ise duz metin olarak
veriyordu. Istemcinin tek bir sekle bakabilmesi icin hepsi burada ayni govdeye
cevriliyor.

Ikinci islevi guvenlik: beklenmeyen bir istisna kullaniciya yigin izi, ic hata
mesaji veya kaynak dosya yolu olarak sizmaz; yalnizca sabit bir kod ve kisa bir
mesaj doner.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
# Starlette'in HTTPException'i kayit edilir, FastAPI'ninki onun alt sinifi:
# yonlendirme hatalari (404/405) Starlette tarafindan atildigi icin yalnizca
# FastAPI alt sinifina kayit yapmak onlari kacirir.
from starlette.exceptions import HTTPException

from rag_tr.contracts import ErrorCode

#: Dogrulama hatasi mesajinin ustune cikabilecegi uzunluk siniri.
_MAX_MESSAGE_CHARS = 200

_STATUS_CODES = {
    404: ErrorCode.NOT_FOUND,
    405: ErrorCode.METHOD_NOT_ALLOWED,
}


def _body(code: ErrorCode, message: str) -> dict:
    return {"detail": {"code": code.value, "message": message}}


def _summarise_validation(exc: RequestValidationError) -> str:
    """Pydantic hatalarini kisa, insan-okunur tek bir cumleye indirger.

    Ham hata listesi girdiyi ve ic alan yollarini yansittigi icin oldugu gibi
    dondurulmez; yalnizca hangi alanin neden reddedildigi yazilir."""
    parts: list[str] = []
    for error in exc.errors()[:3]:
        location = ".".join(str(piece) for piece in error.get("loc", ()) if piece != "body")
        reason = str(error.get("msg", "geçersiz değer"))
        parts.append(f"{location or 'gövde'}: {reason}" if location else reason)
    message = "İstek doğrulanamadı — " + "; ".join(parts) if parts else "İstek doğrulanamadı."
    return message[:_MAX_MESSAGE_CHARS]


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_body(ErrorCode.INVALID_REQUEST, _summarise_validation(exc)),
        )

    @app.exception_handler(HTTPException)
    async def _http(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail
        # Kendi uc noktalarimiz zaten dogru sekli uretiyor; oldugu gibi gecer.
        if isinstance(detail, dict) and "code" in detail:
            return JSONResponse(status_code=exc.status_code, content={"detail": detail},
                                headers=getattr(exc, "headers", None))
        code = _STATUS_CODES.get(exc.status_code, ErrorCode.HTTP_ERROR)
        message = detail if isinstance(detail, str) and detail else "İstek işlenemedi."
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(code, str(message)[:_MAX_MESSAGE_CHARS]),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Hata detayi KASITLI olarak dondurulmez: yigin izi, ic mesaj ve dosya
        # yolu istemciye sizmamali. Ayrinti sunucu loglarinda kalir.
        return JSONResponse(
            status_code=500,
            content=_body(
                ErrorCode.INTERNAL_ERROR,
                "Beklenmeyen bir sunucu hatası oluştu.",
            ),
        )
