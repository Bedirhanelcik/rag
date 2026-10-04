"""Yazma uclarinin yetkilendirmesi.

Tek bir kural var ve tek bir yerde duruyor: korpusu DEGISTIREN her uc nokta
`INGEST_API_TOKEN` ile korunur. Okuma uclari (`/health`, `/documents`,
`/agent/ask`) herkese aciktir -- bunlar icerik uretir ya da sayar, hicbirini
degistirmez.

Neden ayri bir modul: bu kontrol once yalnizca `/ingest`te vardi, `/upload` ve
dokuman silme ise *yalnizca* `UPLOAD_ENABLED` bayragina bakiyordu -- yani
bayrak acik oldugu anda o uclara KIM OLURSA OLSUN yazabiliyordu. Yetki ile
ortam yetenegi ayni bayrakta karismisti. Artik ikisi ayri:

  * YETKI (kim yazabilir)      -> `INGEST_API_TOKEN`, bu modul, tum yazma uclari.
  * ORTAM YETENEGI (acik mi)   -> `UPLOAD_ENABLED`, yalnizca `/upload`, kendi
                                  kodu (`upload_disabled`) ile. Token'dan SONRA
                                  bakilir; paylasimli bir dagitimi gecici olarak
                                  salt okunur yapmak icin duruyor.

Yeni bir kimlik dogrulama sistemi EKLENMEDI: ayni token, ayni baslik bicimi,
ayni hata kodlari.
"""

import secrets

from fastapi import HTTPException, Request

from rag_tr.contracts import ErrorCode


def _error(code: ErrorCode, message: str) -> dict:
    return {"code": code.value, "message": message}


def has_write_access(request: Request, settings) -> bool:
    """Bu ISTEK korpusu degistirebilir mi? Hata firlatmaz, yalnizca yanit verir.

    `require_write_token` ile ayni kurali kullanir; ayri bir predicate olarak
    duruyor cunku `/documents` listesindeki `can_modify` alaninin de ayni
    cevaba ihtiyaci var. Sunucunun token'i olup olmadigina DEGIL, cagiranin
    gecerli bir token sunup sunmadigina bakar: aksi halde token tanimli bir
    dagitimda anonim bir tarayiciya da `can_modify: true` denir, kullaniciya
    silme dugmesi gosterilir ve dugme 401 ile doner.
    """
    expected = getattr(settings, "ingest_api_token", None)
    if not expected:
        return False

    header = request.headers.get("authorization", "")
    scheme, _, supplied = header.partition(" ")
    if scheme.lower() != "bearer":
        return False
    return secrets.compare_digest(supplied.strip(), expected)


def require_write_token(request: Request, settings) -> None:
    """Korpusu degistiren uclar icin `Authorization: Bearer <token>` bekler.

    Token sunucuda hic tanimli degilse uc nokta kapalidir (503): boylece
    yapilandirilmamis bir dagitimda korpusa kimse yazamaz. Tanimliysa baslik
    beklenir ve karsilastirma zamanlama sizintisina kapali yapilir.
    """
    expected = getattr(settings, "ingest_api_token", None)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=_error(
                ErrorCode.INGEST_DISABLED,
                "Yazma uç noktaları kapalı. Sunucu tarafında INGEST_API_TOKEN tanımlayın.",
            ),
        )

    if not has_write_access(request, settings):
        raise HTTPException(
            status_code=401,
            detail=_error(ErrorCode.UNAUTHORIZED, "Geçersiz veya eksik API token'ı."),
        )
