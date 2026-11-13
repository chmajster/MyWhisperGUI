# WhysperGUI

Lokalny panel NiceGUI do transkrypcji audio i wideo za pomocą Faster-Whisper.
Przetwarzanie odbywa się na komputerze użytkownika; aplikacja nie wysyła nagrań
do zewnętrznego API. Dostęp do sieci jest potrzebny przy instalacji i pobieraniu
modeli. Po ich pobraniu transkrypcja i VAD działają offline.

## Wymagania i uruchomienie

- Windows 10/11, 64-bitowy Python 3.11 lub 3.12 i odpowiednia ilość wolnego miejsca.
- Do CUDA: aktualny sterownik NVIDIA, zgodny z kartą; na komputerze docelowym
  sprawdzono RTX 5070 Ti, compute capability 12.0 i sterownik 621.14.
- Instalator pobiera m.in. duży pakiet PyTorch CUDA oraz model `tiny` do testu.
  Model domyślny `large-v3` zajmuje około 3 GB i jest pobierany przy pierwszym
  użyciu lub przez zakładkę **Modele AI**.

Dwukrotnie kliknij `start.bat`. Jeśli środowiska jeszcze nie ma, skrypt uruchomi
instalator, a następnie otworzy `http://127.0.0.1:8080`. Pierwsza instalacja może
potrwać. Aplikację zamyka się przez Ctrl+C w jej oknie konsoli. Alternatywnie:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\start.bat
```

Instalator korzysta z istniejącego `.venv` albo tworzy nowe środowisko. Nie zmienia
globalnego Pythona ani CUDA Toolkit. Wykonuje `pip check`, diagnostykę i test
inferencji modelu `tiny`. W razie błędu podaje raport; nie stwierdza sukcesu GPU
na podstawie samego wykrycia karty.

## Repozytorium Git

Repozytorium zawiera kod, skrypty instalacyjne, listę zależności oraz małe próbki
potrzebne do testów. `.gitignore` pomija lokalne środowisko `.venv`, domyślne
modele, nagrania, historię, eksporty i logi. Pozostają one na komputerze i nie
muszą być wysyłane do GitHuba. Po sklonowaniu `start.bat` utworzy środowisko i
pobierze zależności oraz model używany do testu instalacji.

Jeśli w ustawieniach wybierzesz własny katalog modeli lub eksportów, umieść go
poza repozytorium albo dodaj go do `.gitignore`. Przed każdym commitem sprawdź
`git status`, szczególnie pod kątem plików prywatnych i dużych plików.

Po utworzeniu pustego repozytorium na GitHubie i ustawieniu go jako `origin`:

```powershell
git add .
git status --short
git commit -m "Initial commit"
git push -u origin main
```

## Zastosowany stos CUDA

Wersje podstawowych składników: Faster-Whisper 1.2.1, CTranslate2 4.8.4,
PyTorch 2.9.1 z oficjalnego repozytorium `cu128`, CUDA 12.8 oraz cuDNN 9 z
pakietu PyTorch. Aplikacja dodaje katalog DLL pakietu PyTorch do ścieżki procesu
przed załadowaniem CTranslate2. Globalny CUDA Toolkit nie jest wymagany.
W razie problemów sprawdź również instalację Microsoft Visual C++ Redistributable
x64. Potrzebny jest sterownik obsługujący daną kartę i runtime.

Na RTX 5070 Ti w tym środowisku wykonano rzeczywistą inferencję `float16` i
`int8_float16` w CTranslate2 4.8.4. Raportowana dostępność precyzji nie gwarantuje,
że inna wersja CTranslate2 będzie działać identycznie na Blackwell: po aktualizacji
należy ponowić test. Tryb automatyczny dobiera `float16` albo `int8_float16` na
podstawie dostępnej VRAM i możliwości silnika. Są to orientacyjne progi, a nie
gwarancja, że model zmieści się przy każdym obciążeniu. W przypadku błędu pamięci
zamknij inne aplikacje GPU lub wybierz mniejszy model.

Źródła: [Faster-Whisper — wymagania GPU](https://github.com/SYSTRAN/faster-whisper#gpu),
[CTranslate2 — instalacja Windows](https://opennmt.net/CTranslate2/installation.html),
[lista zmian CTranslate2](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md)
i [oficjalne pakiety PyTorch CUDA 12.8](https://pytorch.org/get-started/previous-versions/).

## Jak korzystać

1. W **Modele AI** pobierz model albo pozwól aplikacji pobrać go przy pierwszym
   zadaniu. Pasek pobierania opiera się na rozmiarach otrzymanych plików.
2. W **Transkrypcja** przeciągnij jeden lub więcej plików, wybierz model i język,
   a następnie dodaj zadania do kolejki. Obsługiwane są MP3, WAV, FLAC, M4A,
   AAC, OGG, OPUS, MP4, MKV, MOV, AVI i WEBM. PyAV wydobywa audio z wideo.
3. W **Kolejka** można zmieniać kolejność oczekujących zadań, anulować je i
   ponawiać po błędzie. W trakcie inferencji postęp oznacza przetworzony czas
   nagrania; przy nieznanym czasie trwania procent nie jest zgadywany. Przycisk
   **Podgląd na żywo** pokazuje dotychczas rozpoznane segmenty i szacowany czas
   do końca (ETA). ETA opiera się na zmierzonym tempie transkrypcji, jest
   orientacyjne i pojawia się dopiero po przetworzeniu części nagrania.
4. Po zakończeniu zadania otwiera się edytor z odtwarzaczem, segmentami,
   automatycznym zapisem, wyszukiwaniem i pobieraniem TXT, SRT, VTT, JSON i CSV.
   **Ponów eksport** nadpisuje pliki eksportu poprawionym tekstem. Korekta tekstu
   segmentu usuwa wcześniejsze znaczniki słów tego segmentu, bo mogłyby już nie
   odpowiadać treści.
5. **Nagrywanie** zapisuje wybrany lokalny mikrofon do WAV. Po zakończeniu można
   odsłuchać nagranie, pobrać je lub przekazać do kolejki.

Wybór `GPU · CUDA` jest domyślny. Brak CUDA powoduje błąd z podpowiedzią —
przełączenie na CPU wymaga świadomego wyboru w formularzu. Domyślnie model po
zadaniu jest zwalniany; opcja **Pozostaw model w pamięci** pozwala szybciej
wykonywać następne zadania kosztem VRAM. Jednocześnie działa najwyżej jedna
transkrypcja GPU; limit wszystkich pracowników można ustawić na 1 lub 2.

Tryb tłumaczenia Whisper tłumaczy mowę na **angielski**. Model `turbo` nie jest
przeznaczony do tłumaczenia i jego połączenie z tym trybem jest odrzucane.
Osobny silnik tłumaczenia na polski ani WASAPI loopback nie są dołączone.

## Dane, ograniczenia i bezpieczeństwo

- Historia i ustawienia: `data/whysper.sqlite3`; importy: `data/uploads`;
  nagrania: `data/recordings`; dziennik: `logs/app.log`; domyślne modele:
  `models`; eksporty: `output`. Katalogi modeli i eksportu można zmienić w GUI.
- Pliki eksportowane są pod identyfikatorem zadania. Usunięcie wpisu historii
  nie usuwa istniejących plików eksportu. Niepowiązane importy i pliki tymczasowe
  starsze niż 7 dni są sprzątane przy starcie; importy powiązane z historią i
  nagrania są zachowywane.
- Serwer nasłuchuje tylko na `127.0.0.1:8080` i odrzuca obce nagłówki `Host` i
  `Origin`. Nie udostępniaj go w sieci za pomocą zewnętrznego proxy bez własnej
  ochrony dostępu.
- Nagranie jest dekodowane do maksymalnie 30-sekundowych porcji, co ogranicza
  zużycie RAM. Granica porcji może przeciąć wypowiedź i pogorszyć rozpoznanie
  na jej styku. Automatyczny język jest ustalany na podstawie pierwszej porcji.
  Anulowanie inferencji następuje przy najbliższej granicy segmentu lub porcji;
  trwające pobieranie modelu musi się najpierw zakończyć.
- `RTF` to czas zadania podzielony przez długość nagrania; czas obejmuje również
  ładowanie modelu i przygotowanie. Mniejsza wartość oznacza szybsze działanie.
- Nagrywanie wymaga działającego wejścia PortAudio w systemie, a odtwarzanie
  wideo zależy od kodeków obsługiwanych przez przeglądarkę.

## Diagnostyka i utrzymanie

**Diagnostyka → Uruchom test CUDA** ładuje model `tiny` i wykonuje prawdziwą
inferencję w osobnym procesie. Pokaże też błędy brakujących DLL. W konsoli:

```powershell
.\.venv\Scripts\python.exe -m app.core.gpu_manager --test
powershell -ExecutionPolicy Bypass -File .\repair.ps1
powershell -ExecutionPolicy Bypass -File .\repair.ps1 -Full
powershell -ExecutionPolicy Bypass -File .\update.ps1
```

Zwykła naprawa instaluje brakujące zależności i wykonuje test; `-Full` wymusza
ponowną instalację pakietów oraz dużego pakietu PyTorch. Aktualizacja źródeł
przez `update.ps1` działa wyłącznie w czystym checkoutcie Git i używa
`git pull --ff-only`; w innym przypadku należy zaktualizować źródła osobno.
`uninstall.ps1` usuwa `.venv`, zachowując dane i kod źródłowy. Jawne
`uninstall.ps1 -RemoveData` usuwa także lokalne `data`, `models`, `output` i
`logs`; nie usuwa niestandardowych katalogów wskazanych w ustawieniach.

## Testy

```powershell
.\.venv\Scripts\python.exe -m pytest -q -m "not gpu and not large_gpu and not ui"
$env:WHYSPER_GPU_TEST = '1'
.\.venv\Scripts\python.exe -m pytest -q -m gpu
$env:WHYSPER_LARGE_TEST = '1'
.\.venv\Scripts\python.exe -m pytest -q -m large_gpu
# Przy uruchomionym serwerze:
$env:WHYSPER_UI_TEST = '1'
.\.venv\Scripts\python.exe -m pytest -q -m ui
# Pełny przebieg GUI, w tym GPU, edycja i pobranie TXT:
$env:WHYSPER_UI_FULL_TEST = '1'
.\.venv\Scripts\python.exe -m pytest -q -m ui
```

Test GPU wymaga faktycznej RTX 5070 Ti i pobranego `tiny`: wykonuje inferencję
CUDA na nagraniu mowy i sprawdza wynik, język, kolejkę, historię i eksport.
Test `large_gpu` dodatkowo tworzy MP3 z polskiego nagrania CC0 i uruchamia pełne
zadanie `large-v3` na CUDA. Test `ui` sprawdza zdarzenia NiceGUI i wysłanie
lokalnej próbki do aplikacji; nie zastępuje wizualnego testu w przeglądarce ani
testu fizycznego mikrofonu. Pełny wariant testu GUI tworzy własne zadanie GPU;
uruchamiaj go wtedy, gdy nie ma ważnej transkrypcji w toku.
