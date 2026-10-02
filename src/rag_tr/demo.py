"""Demo fixture'larinin dogrulama katmani.

Web arayuzu Gemini kotasi harcamadan gercek agent davranisini gosterebilsin diye
onceki kosulardan turetilmis ornekler `demo/fixtures.json` icinde saklanir.

Iki tasarim karari:

1. Fixture govdesi P0'daki HTTP semalarinin ta kendisidir (`AgentSourceItem`,
   `AgentStepItem`, `AgentToolCallItem`, `AgentStatus`). Boylece demo verisi ile
   canli `/agent/ask` yaniti ayni sekle sahip olur ve arayuz tek bir render
   yolu kullanir -- format ikiye ayrilamaz.

2. Butun modeller `extra="forbid"` ile tanimlanir. Bu, dusunce zinciri
   (chain-of-thought) veya baska gizli bir alanin fixture dosyasina sessizce
   girmesini imkansiz kilar: bilinmeyen alan dogrulamada hata verir.

Bu modul hicbir LLM saglayicisini import etmez ve hicbir istemci kurmaz; demo
verisini okumak tamamen yereldir.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag_tr.agent.contracts import AgentStatus
from rag_tr.api.agent_schemas import (
    AgentAskResponse,
    AgentSourceItem,
    AgentStepItem,
    AgentToolCallItem,
)

# Proje kokune gore cozulur (data/uploads, data/chroma ile ayni konvansiyon);
# paket --no-editable kuruldugunda __file__ site-packages'i gosterecegi icin
# paket yoluna gore cozulmez.
DEFAULT_FIXTURES_PATH = Path("demo/fixtures.json")

ProvenanceSource = Literal["live_smoke_test", "live_eval_suite", "handcrafted"]


class FixtureProvenance(BaseModel):
    """Hangi alanlarin gercekten gozlemlendigini, hangilerinin agent'in
    deterministik biciminden yeniden kuruldugunu kayda gecer."""

    model_config = ConfigDict(extra="forbid")

    source: ProvenanceSource
    note: str
    reconstructed_fields: list[str] = Field(default_factory=list)


class DemoFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    # Canli sonuc gibi gosterilmemesi icin zorunlu ve makine-okunur.
    synthetic: bool
    provenance: FixtureProvenance

    answer: str = Field(min_length=1)
    status: AgentStatus
    sources: list[AgentSourceItem]
    steps: list[AgentStepItem]
    tool_calls: list[AgentToolCallItem]

    def to_response(self) -> AgentAskResponse:
        """Canli uc noktanin dondurdugu semanin aynisi."""
        return AgentAskResponse(
            answer=self.answer,
            status=self.status,
            sources=self.sources,
            steps=self.steps,
            tool_calls=self.tool_calls,
        )


class DemoFixtureFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Dosya seviyesinde de isaretli: tuketen taraf bunun demo oldugunu
    # tek bir alana bakarak anlayabilir.
    mode: Literal["demo"]
    note: str
    fixtures: list[DemoFixture] = Field(min_length=1)

    def by_id(self, fixture_id: str) -> DemoFixture | None:
        for fixture in self.fixtures:
            if fixture.id == fixture_id:
                return fixture
        return None

    def by_question(self, question: str) -> DemoFixture | None:
        """Sorulari normalize ederek eslestirir (bosluk/buyuk-kucuk harf farki).

        Uyari: Python Turkce locale kullanmaz, bu yuzden noktasiz "i" iceren bir
        metnin buyuk harfe cevrilip geri casefold edilmesi ayni dizeyi vermez
        ("dagi" -> "DAGI" -> "dagi"). Bu nedenle eslestirme kullanicinin yazdigi
        metin uzerinde yapilmali, buyuk harfe cevrilmis bir kopya uzerinde
        degil."""
        wanted = " ".join(question.split()).casefold()
        for fixture in self.fixtures:
            if " ".join(fixture.question.split()).casefold() == wanted:
                return fixture
        return None


def load_fixtures(path: str | Path | None = None) -> DemoFixtureFile:
    """Fixture dosyasini okur ve dogrular. Ag veya API erisimi gerektirmez."""
    target = Path(path) if path is not None else DEFAULT_FIXTURES_PATH
    return DemoFixtureFile.model_validate_json(target.read_text(encoding="utf-8"))
