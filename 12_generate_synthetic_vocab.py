#!/usr/bin/env python3
"""
Generiert themenbasierte synthetische Trainingssätze für unterrepräsentierte Domänen.

Im Gegensatz zu 08_generate_synthetic.py (generischer Handy-Stil) zielt dieses
Skript auf spezifische Themengebiete mit maßgeschneiderten Prompts — Geschichte,
Medizin, Technik, Natur usw. Jedes Thema hat einen eigenen Kontext und Stil.

Output: data/synthetic_themen.txt  (separater File, nicht synthetic_de.txt)

Usage:
  .venv_ml/bin/python 12_generate_synthetic_vocab.py [--per-topic 50]
  .venv_ml/bin/python 12_generate_synthetic_vocab.py --topics mittelalter,medizin
  .venv_ml/bin/python 12_generate_synthetic_vocab.py --list-topics
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")

# ── Themen mit spezifischen Prompts ──────────────────────────────────────────
# Jedes Thema: name, beschreibung, system_zusatz, beispiele
# system_zusatz ergänzt den allgemeinen System-Prompt für das Thema

TOPICS = {
    "historia": {
        "name": "Historia & Średniowiecze",
        "beschreibung": (
            "Zdania o historii, średniowieczu, zamkach, wydarzeniach historycznych — "
            "tak jak ludzie piszą w rozmowach, na czacie czy w Googlach. "
            "Nie styl encyklopedyczny, tylko naturalna rozmowa."
        ),
        "beispiele": [
            "Byliśmy wczoraj na zamku w Malborku, było niesamowicie.",
            "Kiedy właściwie upadło powstanie warszawskie?",
            "Historia Polski jest strasznie skomplikowana.",
            "Mieszko I przyjął chrzest w 966 roku, prawda?",
            "Moja babcia opowiadała o wojnie, straszne rzeczy.",
            "Kopernik był Polakiem, nie Niemcem!",
            "W średniowieczu ludzie naprawdę wierzyli w smoki?",
            "Jan III Sobieski uratował Europę pod Wiedniem.",
        ],
    },
    "medycyna": {
        "name": "Medycyna & Zdrowie",
        "beschreibung": (
            "Zdania o zdrowiu, wizytach u lekarza, diagnozach, leczeniu — "
            "tak jak mówi się na co dzień. Rozmowy ze znajomymi, rodziną, "
            "umawianie wizyt, opisywanie objawów."
        ),
        "beispiele": [
            "Muszę iść do dermatologa, bo mam wysypkę.",
            "Morfologia wyszła całkiem nieźle, na szczęście.",
            "Lekarz zapisał mi antybiotyk na zapalenie oskrzeli.",
            "Mam nadciśnienie, muszę brać leki codziennie.",
            "Znacie dobrego kardiologa w Warszawie?",
            "Rezonans magnetyczny nic nie wykazał, dzięki Bogu.",
            "Biorę metforminę na cukrzycę od dwóch lat.",
            "Rehabilitacja po operacji kolana bardzo pomaga.",
        ],
    },
    "technologia": {
        "name": "Komputery & Technologia",
        "beschreibung": (
            "Zdania o komputerach, software, internecie, AI, grach — "
            "codzienne rozmowy o technice, pytania, problemy. "
            "Nie tylko fachowy żargon, ale jak zwykli ludzie mówią o technologii."
        ),
        "beispiele": [
            "Mój laptop znowu się zawiesił, nie mam pojęcia czemu.",
            "Próbowałeś już nowego ChatGPT? Podobno jest lepszy.",
            "Blockchain dalej jest dla mnie czarną magią.",
            "Linux jest za trudny dla przeciętnego użytkownika.",
            "Jaką kartę graficzną polecasz do gier?",
            "Nie ufam cloud synchronizacji, boję się o prywatność.",
            "Nowy iPhone ma ponoć lepszy procesor.",
            "Uczę się Pythona i całkiem mi to wchodzi.",
        ],
    },
    "natura": {
        "name": "Natura & Środowisko",
        "beschreibung": (
            "Zdania o naturze, zwierzętach, roślinach, ochronie środowiska, "
            "klimacie — codzienne rozmowy: na spacerze, w newsach, "
            "o zwierzętach domowych, ogrodzie, pogodzie."
        ),
        "beispiele": [
            "Widzieliśmy dzisiaj bociana na łące, piękny widok!",
            "Ten dąb w naszym ogrodzie ma chyba sto lat.",
            "Populacje pszczół dramatycznie maleją, to straszne.",
            "Oglądałeś film o rafach koralowych?",
            "Kompostowanie wcale nie jest takie trudne.",
            "Te fale upałów są coraz gorsze przez zmiany klimatu.",
            "W parku narodowym nie wolno dotykać roślin.",
            "To lato jest zdecydowanie za suche dla rolnictwa.",
        ],
    },
    "polityka": {
        "name": "Polityka & Społeczeństwo",
        "beschreibung": (
            "Zdania o polityce, wyborach, partiach, debatach społecznych — "
            "codzienne rozmowy w rodzinie, ze znajomymi. Żadnej propagandy, "
            "normalny dyskurs."
        ),
        "beispiele": [
            "Nie ogarniam tego rządu, ciągle się kłócą.",
            "Frekwencja w wyborach była zaskakująco wysoka.",
            "Widziałeś przemówienie prezydenta wczoraj?",
            "System emerytalny musi być pilnie reformowany.",
            "Podoba mi się nowy program mieszkaniowy.",
            "Unia Europejska robi czasem naprawdę dobre rzeczy.",
            "Polityka klimatyczna jest ważna, ale wykonanie kuleje.",
            "Nowy prezydent miasta ma ciekawe pomysły.",
        ],
    },
    "muzyka": {
        "name": "Muzyka & Koncerty",
        "beschreibung": (
            "Zdania o muzyce, koncertach, festiwalach, zespołach, "
            "instrumentach — jak fani piszą o muzyce na co dzień. "
            "Rekomendacje, wrażenia z koncertów, dyskusje o albumach."
        ),
        "beispiele": [
            "Koncert wczoraj był absolutnie legendarny!",
            "Kult na żywo to zupełnie inne przeżycie.",
            "Uczę się grać na pianinie, jestem totalnym początku.",
            "Nowa płyta Dawida Podsiadło jest świetna.",
            "Festiwal w Opolu mam na liście marzeń.",
            "Muzyka klasyczna brzmi elitarnie ale pięknie.",
            "Macie już bilety na Open'er?",
            "Filharmonia Narodowa jest po prostu niezastąpiona.",
        ],
    },
    "sport": {
        "name": "Sport & Ruch",
        "beschreibung": (
            "Zdania o sporcie — piłka nożna, bieganie, rower, siłownia, "
            "pływanie. Jak ludzie rozmawiają o treningu, zawodach, "
            "klubach sportowych."
        ),
        "beispiele": [
            "Półmaraton był ciężki, ale dałem radę!",
            "Kiedy macie trening w przyszłym tygodniu?",
            "Moje kolano dalej boli przy bieganiu.",
            "Liga Mistrzów wczoraj była rozczarowująca.",
            "Zatrudniłem trenera personalnego, warto.",
            "Sklepka górska wymaga dobrego obuwia.",
            "Triathlon to moje następne wielkie wyzwanie.",
            "Stadion był pełny, super atmosfera.",
        ],
    },
    "jedzenie": {
        "name": "Jedzenie & Gotowanie",
        "beschreibung": (
            "Zdania o przepisach, restauracjach, gotowaniu, diecie — "
            "codzienne rozmowy o jedzeniu, rekomendacje restauracji, "
            "zmiany w diecie, regionalne specjały."
        ),
        "beispiele": [
            "Zrobiłem wczoraj pierogi od zera, wyszły niezłe.",
            "Znacie dobrą wegańską restaurację w okolicy?",
            "Pierogi od babci są zawsze najlepsze.",
            "Próbuję diety bezglutenowej od tygodnia.",
            "Przepis na sernik wcale nie jest taki trudny.",
            "Nietolerancja laktozy strasznie komplikuje zakupy.",
            "W sobotę jemy zawsze śniadanie z rodziną.",
            "Schabowy z ziemniakami to klasyka polskiej kuchni.",
        ],
    },
    "media": {
        "name": "Książki, Filmy & Muzyka",
        "beschreibung": (
            "Zdania o książkach, filmach, serialach, podcastach — "
            "jak ludzie rozmawiają o mediach: rekomendacje, wrażenia, "
            "opinie. Naturalnie, nie katalogowo."
        ),
        "beispiele": [
            "Skończyłem czytać książkę w zeszłym tygodniu, była super.",
            "Słuchałeś tego audiobooka? Fajna sprawa.",
            "Rzadko kupuję już płyty, wszystko jest w streamingu.",
            "Ten film na Netflixie jest naprawdę dobry.",
            "Serial polecił mi znajomy, leży od miesięcy.",
            "Jakość obrazu na Blu-ray robi wrażenie.",
            "Wolę słuchać podcastów niż muzyki przy gotowaniu.",
            "To najlepsza książka jaką przeczytałem w tym roku.",
        ],
    },
    "dom": {
        "name": "Majsterkowanie & Dom",
        "beschreibung": (
            "Zdania o majsterkowaniu, naprawach, urządzaniu — "
            "codzienne rozmowy: zakupy materiałów, narzędzia, "
            "projekty DIY, pomoc sąsiedzka."
        ),
        "beispiele": [
            "Masz gwoździe? Chcę powiesić obraz.",
            "Potrzebuję dłuższych wkrętów, te nie pasują.",
            "Kołek jest za duży do tej dziury.",
            "Pożyczysz mi wiertarkę na chwilę?",
            "Bosch robi naprawdę dobre elektronarzędzia.",
            "Muszę przykręcić półkę do ściany.",
            "Kran kapie od tygodnia, wezwę hydraulika.",
            "Myślisz że M8 czy M10 będą lepsze?",
        ],
    },
    "lazienka": {
        "name": "Łazienka & Remont",
        "beschreibung": (
            "Zdania o łazience — remont, armatura, wykończenie. "
            "Jak ludzie rozmawiają o planowaniu łazienki, naprawach "
            "i produktach łazienkowych na co dzień."
        ),
        "beispiele": [
            "Wanna jest tak stara, że trzeba ją wymienić.",
            "Myślimy o prysznicu na podłodze, bez brodzika.",
            "Fugi w kabinie prysznicowej są zapleśniałe.",
            "Kran cieknie już od tygodni.",
            "Nowe płytki odmienią tę łazienkę.",
            "Chcemy zrobić generalny remont łazienki.",
            "Hydraulik powiedział, że trzy tygodnie czekania.",
            "Kabina prysznicowa jest strasznie trudna do czyszczenia.",
        ],
    },
    "mieszkanie": {
        "name": "Mieszkanie & Meble",
        "beschreibung": (
            "Zdania o urządzaniu, meblach, przeprowadzce — "
            "codzienne rozmowy o nowych meblach, remoncie, "
            "przeprowadzce, IKEA, podłogach i kolorach."
        ),
        "beispiele": [
            "Potrzebujemy jeszcze szafy do sypialni.",
            "Składam szafkę z IKEA, ostatni raz!",
            "Przeprowadzka była koszmarem, trzy kartony się rozpadły.",
            "Znacie dobry i niedrogi laminat?",
            "Ta komoda pasuje idealnie pod ścianę w przedpokoju.",
            "Myślimy o malowaniu salonu na nowo.",
            "Kanapa jest już tak wygnieciona, trzeba wymienić.",
            "Regał z litego drewna wygląda przepięknie.",
        ],
    },
    "zakupy": {
        "name": "Zakupy & Supermarket",
        "beschreibung": (
            "Zdania o zakupach — supermarkety, targ, ceny, promocje. "
            "Jak ludzie mówią o codziennych zakupach: listy, "
            "rozmowy przy kasie, porównywanie cen."
        ),
        "beispiele": [
            "Muszę skoczyć do sklepu, masz listę?",
            "Znowu zapomniałem listy zakupów w domu.",
            "Kupisz mleko po drodze?",
            "Na targu warzywa są o wiele świeższe.",
            "Kolejki do kasy w sobotę są okropne.",
            "Staramy się kupować mniej plastiku.",
            "Dyskont jest tańszy, ale brak w nim bio warzyw.",
            "Kupujemy chleb u lokalnego piekarza zamiast w supermarkecie.",
        ],
    },
    "auto": {
        "name": "Auto & Warsztat",
        "beschreibung": (
            "Zdania o samochodzie — warsztat, przegląd, wymiana opon, "
            "awarie, paliwo. Jak kierowcy rozmawiają o swoich pojazdach: "
            "naprawy, koszty, porady."
        ),
        "beispiele": [
            "Muszę jechać na przegląd, już po terminie.",
            "Tarcze hamulcowe trzeba wkrótce wymienić.",
            "Wymiana opon na zimowe już zamówiona w warsztacie.",
            "Warsztat znowu policzył za dużo.",
            "Przebieg ponad 200 tysięcy, ale silnik chodzi świetnie.",
            "Dolałeś już oleju silnikowego?",
            "Klimatyzacja dziwnie hałasuje od zeszłego lata.",
            "Zmieniam felgi samemu, to wcale nie jest trudne.",
        ],
    },
    "ubrania": {
        "name": "Ubrania & Moda",
        "beschreibung": (
            "Zdania o ubraniach, modzie, zakupach odzieżowych — "
            "codzienne rozmowy o zakupie, rozmiarach, jakości, "
            "markach, praniu i stylu."
        ),
        "beispiele": [
            "Potrzebuję kurtkę na zimę, jaką polecasz?",
            "Te spodnie niestety już na mnie nie leżą.",
            "Jaki rozmiar nosisz normalnie?",
            "Koszula skurczyła się po pierwszym praniu.",
            "Wolę kupować ubrania stacjonarnie niż online.",
            "Zamek się zepsuł, typowa chińska jakość.",
            "Jesień to moja ulubiona pora roku modowo.",
            "Te buty są przecenione, jeszcze myślę.",
        ],
    },
    "logistyka": {
        "name": "Paczki & Wysyłka",
        "beschreibung": (
            "Zdania o paczkach, listach, wysyłce i dostawie — "
            "od zwykłych paczek po profesjonalną logistykę: "
            "spedycja, odprawa celna, magazyn, import/eksport."
        ),
        "beispiele": [
            "Paczka już trzy dni w drodze, śledzenie nie pomaga.",
            "Poczta Polska znowu zgubiła paczkę.",
            "Co muszę zgłosić do cła jak zamawiam z USA?",
            "Kurier podał zły termin, teraz czekam kolejny dzień.",
            "Odprawa celna z Chin czasem trwa wieczność.",
            "Dostałem awizo mimo że byłem w domu.",
            "Łańcuch dostaw przez strajk był całkiem zerwany.",
            "Możesz nadać jako ekspres? Potrzebuję na jutro.",
        ],
    },
    "rodzina": {
        "name": "Rodzina & Dzieci",
        "beschreibung": (
            "Zdania o rodzinie, dzieciach, wychowaniu — "
            "od pieluch po nastolatki. Codzienne rozmowy rodziców, "
            "dziadków, rodzeństwa."
        ),
        "beispiele": [
            "Mały wreszcie przesypia noce, ulga!",
            "Mama dzisiaj zajmuje się dziećmi, mamy wieczór dla siebie.",
            "Jak wytłumaczyć dziecku rozwód?",
            "Przedszkole znowu zamknięte, oszaleję.",
            "Z nastolatkami rozmowa bywa bardzo trudna.",
            "Myślimy o drugim dziecku, ale boimy się.",
            "Teściowa wtrąca się w wychowanie, nie wyrabiam.",
            "Podwyższyć kieszonkowe czy nie? Kłócimy się o to.",
        ],
    },
    "edukacja": {
        "name": "Szkoła & Studia",
        "beschreibung": (
            "Zdania o szkole, studiach, egzaminach, maturze — "
            "z perspektywy uczniów, studentów i rodziców. "
            "Oceny, stres, wakacje, stypendium."
        ),
        "beispiele": [
            "Matematyka na maturze była koszmarna, oblewam.",
            "Kiedy kończy się rekrutacja na studia?",
            "Moje dziecko znowu zapomniało zeszytu.",
            "Stypendium ledwo starcza na życie.",
            "Piszę pracę magisterską, oby do obrony.",
            "Wychowawczyni dzwoniła z powodu zachowania.",
            "Jaki kierunek wybrać? Totalnie nie wiem.",
            "Matura była bardziej stresująca niż myślałem.",
        ],
    },
    "zwiazki": {
        "name": "Znajomości & Randki",
        "beschreibung": (
            "Zdania o przyjaźni, związkach, randkach, aplikacjach — "
            "rozstania, zazdrość, flirt, długie związki, zaufanie. "
            "Jak ludzie naprawdę o tym mówią."
        ),
        "beispiele": [
            "Poznaliśmy się na Tinderze, jesteśmy razem dwa lata.",
            "Kłócę się z najlepszą przyjaciółką, nie wiem co robić.",
            "Nie pisał od trzech dni, chyba go olewam.",
            "Jesteśmy razem dziesięć lat, myślimy o ślubie.",
            "Związki na odległość są strasznie męczące.",
            "Zerwałem, to już nie było to samo.",
            "Randki po czterdziestce są totalnie inne.",
            "Znajomi martwią się o mnie od rozstania.",
        ],
    },
    "praca": {
        "name": "Praca & Biuro",
        "beschreibung": (
            "Zdania o pracy, biurze, współpracownikach, szefie, "
            "spotkaniach, home office, aplikacjach o pracę, "
            "pensji. Jak pracownicy rozmawiają naprawdę."
        ),
        "beispiele": [
            "To spotkanie mogło być mailem.",
            "Szef wymaga żebym był dostępny w weekend.",
            "W home office jestem bardziej produktywny.",
            "Aplikuję na nową posadę, jestem zestresowany.",
            "Koleżanka z biura strasznie mnie irytuje.",
            "Znowu nie będzie podwyżki w tym roku.",
            "Nowa praca lepiej płaci, ale dojazd jest daleko.",
            "Mam dzisiaj ostatni dzień, wreszcie odchodzę.",
        ],
    },
    "finanse": {
        "name": "Finanse & Bankowość",
        "beschreibung": (
            "Zdania o pieniądzach, banku, koncie, kredycie, "
            "ubezpieczeniu, emeryturze, podatkach, inwestowaniu, "
            "oszczędzaniu — jak zwykli ludzie o tym mówią."
        ),
        "beispiele": [
            "Debet znowu na limicie, nie wiem jak to się stało.",
            "Nie rozumiem swojego PIT-u w ogóle.",
            "Inwestować w ETF czy trzymać na koncie oszczędnościowym?",
            "Karta została odrzucona, wstyd przy kasie.",
            "Emerytura z ZUS mi nie wystarczy.",
            "Odkładam 500 zł miesięcznie, idzie opornie.",
            "Opłaty bankowe wzrosły, myślę o zmianie banku.",
            "W końcu dostaliśmy kredyt na mieszkanie.",
        ],
    },
    "zwierzeta": {
        "name": "Zwierzęta Domowe",
        "beschreibung": (
            "Zdania o psach, kotach, ptakach, rybkach — "
            "opieka, weterynarz, karma, wychowanie, "
            "zachowanie, strata."
        ),
        "beispiele": [
            "Pies znowu pogryzł kapcie, nie mam siły.",
            "Kot je tylko droższą karmę, innej nie tknie.",
            "U weterynarza znowu drożej niż myślałem.",
            "Stary pies będzie musiał być uśpiony, to boli.",
            "Gdzie mogę pojechać z psem na wakacje?",
            "Kot i pies w końcu się polubili po roku.",
            "Moja papuga uczy się mówić, jest urocza.",
            "Kto zaopiekuje się kotem na wakacje?",
        ],
    },
    "pogoda": {
        "name": "Pogoda & Pory Roku",
        "beschreibung": (
            "Zdania o pogodzie, porach roku, zmianach klimatu, "
            "burzach, śniegu, upałach — jak Polacy na co dzień "
            "mówią o pogodzie."
        ),
        "beispiele": [
            "Znowu za gorąco jak dla mnie.",
            "Mam nadzieję że lato przyjdzie w tym roku do Polski.",
            "Śnieg wszystko wycisza, uwielbiam to.",
            "Według prognozy jutro wreszcie przestanie padać.",
            "Takich burz kiedyś nie było tak często.",
            "Jesień to moja ulubiona pora roku.",
            "Trzydzieści stopni w cieniu i ani jednej chmury.",
            "Zima w Krakowie jest szara ale jakoś przytulna.",
        ],
    },
    "gry": {
        "name": "Gry Wideo & Gaming",
        "beschreibung": (
            "Zdania o grach, konsolach, PC, grach mobilnych, "
            "streamingu, esporcie — od casuali po hardcore'owych "
            "graczy. Jak gracze naprawdę rozmawiają."
        ),
        "beispiele": [
            "Nowa łatka zepsuła wszystko, jestem wściekły.",
            "Gram godzinami i nie widzę kiedy leci czas.",
            "Jaka karta graficzna się teraz opłaca?",
            "Ta gra ma dwa lata ale dalej jest świetna.",
            "W końcu pokonałem ostatniego bossa po dwudziestu próbach.",
            "Gry mobilne to zwykle naciąganie z mikropłatnościami.",
            "Rodzice nie rozumieją czemu tyle gram.",
            "Nowy season pass znowu kosztuje za dużo.",
        ],
    },
    "hobby": {
        "name": "Hobby & Zainteresowania",
        "beschreibung": (
            "Zdania o hobby: fotografii, ogrodnictwie, szyciu, "
            "malowaniu, majsterkowaniu, zbieractwie, "
            "modelarstwie, gotowaniu jako hobby."
        ),
        "beispiele": [
            "W końcu zebrałem pierwszego własnego pomidora.",
            "Obiektyw jest drogi ale do portretów niezastąpiony.",
            "Robię na drutach sweter, trzecie podejście wychodzi lepiej.",
            "Farby akwarelowe zawsze za bardzo się rozlewają.",
            "Mój projekt modelarski nabiera kształtów.",
            "Na działce jestem po prostu szczęśliwy.",
            "Przepis babci w końcu dopracowałem do perfekcji.",
            "Do makrofotografii potrzebuję statywu.",
        ],
    },
    "podroze": {
        "name": "Podróże & Wakacje",
        "beschreibung": (
            "Zdania o planach podróży, wakacjach, atrakcjach, "
            "transporcie, noclegach — porady, polecenia, "
            "relacje, rezerwacje."
        ),
        "beispiele": [
            "Macie już nocleg na city break?",
            "Hostel w Pradze był zaskakująco fajny.",
            "Autokarem do Krakowa jest taniej niż pociągiem.",
            "Ubezpieczenie turystyczne to podstawa.",
            "Wycieczka z przewodnikiem w Wilnie była super.",
            "Wakacje w camperze byłyby fajną odmianą.",
            "Wiza do Indii trwa wiecznie.",
            "Karta muzealna opłaca się przy dłuższym weekendzie.",
        ],
    },
}

SYSTEM_PROMPT = """Generujesz polskie zdania tak jak prawdziwi ludzie piszą na smartfonie.

Ważne zasady:
- Naturalne i codzienne — nie encyklopedia, nie Wikipedia, nie styl newsowy
- Mieszanka długości: około połowa 6-10 słów, reszta 10-16 słów
- Poprawna pisownia i gramatyka
- Bez cudzysłowów, numeracji, komentarzy
- Dokładnie jedno zdanie na linię, bez pustych linii
- Różnorodność: różne osoby, konteksty, tony
- Każde zdanie zaczyna się wielką literą"""


def build_prompt(topic: dict, n: int) -> str:
    beispiele = "\n".join(topic["beispiele"])
    return (
        f"Temat: {topic['name']}\n"
        f"Kontekst: {topic['beschreibung']}\n\n"
        f"Przykłady poprawnego stylu:\n{beispiele}\n\n"
        f"Napisz teraz dokładnie {n} kolejnych takich zdań na ten temat. "
        f"Urozmaicaj treść, długość i perspektywę. Jedno zdanie na linię."
    )


def ollama_chat(host: str, model: str, system: str, user: str, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model,
        "think": False,
        "stream": False,
        "options": {"num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def parse_sentences(raw: str) -> list[str]:
    # Wenn keine Zeilenumbrüche: auf Satzgrenzen aufsplitten
    if '\n' not in raw and len(raw) > 200:
        raw = re.sub(r'([.!?])([A-ZÄÖÜ])', r'\1\n\2', raw)

    lines = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r'^[\d]+[.)]\s*', '', line)
        line = re.sub(r'^[-•]\s*', '', line)
        if not line or len(line) < 8 or line.endswith(':'):
            continue
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1].strip()
        if line:
            line = line[0].upper() + line[1:]
        lines.append(line)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-topic", type=int, default=50,
                        help="Sätze pro Thema (default: 50)")
    parser.add_argument("--batch", type=int, default=25,
                        help="Sätze pro API-Aufruf (default: 25)")
    parser.add_argument("--model", default="qwen3.6:27b",
                        help="Ollama-Modellname (default: qwen3.6:27b)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama base URL (default: http://localhost:11434)")
    parser.add_argument("--topics", default=None,
                        help="Kommagetrennte Themen-Keys (z.B. mittelalter,medizin)")
    parser.add_argument("--list-topics", action="store_true",
                        help="Verfügbare Themen auflisten und beenden")
    parser.add_argument("--output-dir", default=str(DATA_DIR),
                        help=f"Ausgabeverzeichnis (default: {DATA_DIR}); "
                             "je Thema eine Datei: synthetic_<key>.txt")
    args = parser.parse_args()

    if args.list_topics:
        print("Verfügbare Themen:")
        for key, t in TOPICS.items():
            print(f"  {key:<20} — {t['name']}")
        return

    out_dir = Path(args.output_dir)

    # Themen auswählen
    if args.topics:
        keys = [k.strip().lower() for k in args.topics.split(",")]
        unknown = [k for k in keys if k not in TOPICS]
        if unknown:
            print(f"Unbekannte Themen: {unknown}", file=sys.stderr)
            print(f"Verfügbar: {list(TOPICS.keys())}", file=sys.stderr)
            sys.exit(1)
        selected = {k: TOPICS[k] for k in keys}
    else:
        selected = TOPICS

    total_target = len(selected) * args.per_topic
    print(f"Themen: {len(selected)}  ×  {args.per_topic} Sätze = ~{total_target} Sätze")
    print(f"Modell: {args.model}  →  {out_dir}/synthetic_<thema>.txt\n")

    out_dir.mkdir(parents=True, exist_ok=True)
    total_collected = 0
    t_start = time.time()

    # Zähle bereits vorhandene Sätze pro Topic
    counts = {}
    for key in selected:
        out_path = out_dir / f"synthetic_{key}.txt"
        counts[key] = sum(1 for _ in out_path.open(encoding="utf-8")) if out_path.exists() else 0

    # Round-Robin: eine Batch pro Topic pro Runde bis alle per_topic erreicht
    topic_keys = list(selected.keys())
    errors_per_topic = {k: 0 for k in topic_keys}

    while any(counts[k] < args.per_topic for k in topic_keys):
        for key in topic_keys:
            if counts[key] >= args.per_topic:
                continue
            topic = selected[key]
            out_path = out_dir / f"synthetic_{key}.txt"
            n = min(args.batch, args.per_topic - counts[key])
            pct = int(counts[key] / args.per_topic * 100)
            print(f"  [{key}] {counts[key]}/{args.per_topic} ({pct}%)", end=" ", flush=True)
            try:
                raw = ollama_chat(
                    host=args.host,
                    model=args.model,
                    system=SYSTEM_PROMPT,
                    user=build_prompt(topic, n),
                    max_tokens=n * 40,
                )
                sentences = parse_sentences(raw)
                with out_path.open("a", encoding="utf-8") as f:
                    for s in sentences:
                        f.write(s + "\n")
                counts[key] += len(sentences)
                total_collected += len(sentences)
                errors_per_topic[key] = 0
                print(f"+{len(sentences)}")
            except KeyboardInterrupt:
                print(f"\nAbgebrochen. {total_collected} Sätze gespeichert.")
                return
            except Exception as e:
                print(f"  Fehler: {e}", file=sys.stderr)
                errors_per_topic[key] += 1
                if errors_per_topic[key] > 3:
                    print(f"  Zu viele Fehler bei '{key}', überspringe.")
                    break
                time.sleep(2)

    elapsed = time.time() - t_start
    print(f"\nFertig: {total_collected} Sätze in {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
