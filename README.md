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
należy ponowić test. Tryb automatyczny preferuje `float16` albo `int8_float16` na
podstawie dostępnej VRAM i możliwości wybranej karty; jeśli ich nie obsługuje,
wybiera inną dostępną precyzję CTranslate2. Są to orientacyjne progi, a nie
gwarancja, że model zmieści się przy każdym obciążeniu. W przypadku błędu pamięci
zamknij inne aplikacje GPU lub wybierz mniejszy model.

Źródła: [Faster-Whisper — wymagania GPU](https://github.com/SYSTRAN/faster-whisper#gpu),
[CTranslate2 — instalacja Windows](https://opennmt.net/CTranslate2/installation.html),
[lista zmian CTranslate2](https://github.com/OpenNMT/CTranslate2/blob/master/CHANGELOG.md)
i [oficjalne pakiety PyTorch CUDA 12.8](https://pytorch.org/get-started/previous-versions/).

## Jak korzystać

1. W **Modele AI** pobierz model albo pozwól aplikacji pobrać go przy pierwszym
   zadaniu. Pasek pobierania opiera się na rozmiarach otrzymanych plików.
   Przycisk **Anuluj** zatrzymuje aktywne pobieranie; zachowane pliki mogą być
   wykorzystane przy wznowieniu. Jeśli model jest pobierany dla zadania,
   anulowanie pobierania anuluje również to zadanie, które można później ponowić.
2. W **Transkrypcja** przeciągnij jeden lub więcej plików, wybierz model i język,
   a następnie dodaj zadania do kolejki. Obsługiwane są MP3, WAV, FLAC, M4A,
   AAC, OGG, OPUS, MP4, MKV, MOV, AVI i WEBM. Przy imporcie wideo aplikacja
   zapisuje jego ścieżkę dźwiękową jako FLAC (mono, 16 kHz), sprawdza wynik i
   usuwa lokalną kopię wideo. Oryginał wybrany z dysku pozostaje bez zmian.
   Jeśli konwersja się nie powiedzie, aplikacja zachowa lokalną kopię wideo,
   zgłosi ostrzeżenie i pozwoli użyć jej do transkrypcji.
3. W **Kolejka** można zmieniać kolejność oczekujących zadań, anulować je i
   ponawiać po błędzie. W trakcie inferencji postęp oznacza przetworzony czas
   nagrania; przy nieznanym czasie trwania procent nie jest zgadywany. Przycisk
   **Podgląd na żywo** pokazuje dotychczas rozpoznane segmenty i szacowany czas
   do końca (ETA). ETA opiera się na zmierzonym tempie transkrypcji, jest
   orientacyjne i pojawia się dopiero po przetworzeniu części nagrania.
4. Po zakończeniu zadania otwiera się edytor z odtwarzaczem, segmentami,
   automatycznym zapisem, wyszukiwaniem i pobieraniem TXT, SRT, VTT, JSON i CSV.
   Długie transkrypcje są wyświetlane po 50 fragmentów na stronie; wyszukiwanie,
   kopiowanie i eksport obejmują cały tekst. Wyszukiwanie uwzględnia również
   nazwy rozmówców.
   **Ponów eksport** nadpisuje pliki eksportu poprawionym tekstem. Korekta tekstu
   segmentu usuwa wcześniejsze znaczniki słów tego segmentu, bo mogłyby już nie
   odpowiadać treści. Opcja **Edytuj rozmówców** pozwala poprawić lub wpisać
   nazwę rozmówcy osobno dla każdego fragmentu, także w starszych wynikach.
5. **Nagrywanie** zapisuje wybrany lokalny mikrofon do WAV. Po zakończeniu można
   odsłuchać nagranie, pobrać je lub przekazać do kolejki.
6. W **Napisy na żywo** wybierz mikrofon oraz pobrany model i uruchom napisy.
   Mowa jest transkrybowana w języku wypowiedzi, porcjami po 3–10 sekund.
   Czas oczekiwania obejmuje długość porcji i czas pracy modelu; mniejszy model
   może zmniejszyć opóźnienie. Automatyczny język jest wykrywany osobno dla
   każdej porcji; ręczne wybranie języka może pomóc przy krótkich wypowiedziach.
   Ta funkcja nie tłumaczy jeszcze na inny język.

Wybór `GPU · CUDA` jest domyślny. W polu **Karta GPU** można wskazać jedną z
wykrytych kart NVIDIA CUDA. Wybór w formularzu dotyczy dodawanych zadań;
w zakładce **Ustawienia** można zapisać kartę domyślną. Wybór konkretnej karty
jest zapisywany według UUID, jeśli jest dostępny, dzięki czemu zmiana kolejności
CUDA nie zmienia wybranej karty. Gdy wybrana karta jest niedostępna, aplikacja
zgłasza błąd zamiast uruchomić zadanie na innej. Tryb automatyczny wybiera
pierwszą kartę widoczną dla CUDA. Lista respektuje `CUDA_VISIBLE_DEVICES`.

Brak CUDA powoduje błąd z podpowiedzią — przełączenie na CPU wymaga świadomego
wyboru w formularzu. Domyślnie model po
zadaniu jest zwalniany; opcja **Pozostaw model w pamięci** pozwala szybciej
wykonywać następne zadania kosztem VRAM. Jednocześnie działa najwyżej jedna
transkrypcja GPU; limit wszystkich pracowników można ustawić na 1 lub 2.

Tryb tłumaczenia Whisper tłumaczy mowę na **angielski**. Model `turbo` nie jest
przeznaczony do tłumaczenia i jego połączenie z tym trybem jest odrzucane.
Osobny silnik tłumaczenia na polski ani WASAPI loopback nie są dołączone.

## Parametry transkrypcji

Formularz **Transkrypcja** i zakładka **Ustawienia** zawierają dodatkowe
parametry oraz rozwijaną **Legendę parametrów** z ich opisem. Ustawienia domyślne
można zapisać w zakładce **Ustawienia**; wartości w formularzu transkrypcji
dotyczą zadań dodanych z tego formularza.

Przy temperaturze `0` silnik używa `Beam size` i `patience`. Powyżej `0`
wybiera z losowanych kandydatów `Best of`, a ustawienia wyszukiwania wiązką
nie mają wtedy wpływu. Dostępne są też ograniczenia powtórzeń, próg ciszy
Whisper, podpowiedzi słów i szczegółowe parametry VAD. Wartości VAD mają wpływ
tylko przy włączonym filtrze. Kontekst wcześniejszego tekstu działa w ramach
bieżącej porcji audio, nie pomiędzy niezależnymi porcjami do 30 sekund.

Napisy na żywo korzystają z zapisanych ustawień transkrypcji, w tym filtra VAD
i wybranego GPU. Dźwięk z mikrofonu pozostaje w pamięci i nie jest zapisywany
do pliku. Tekst całej bieżącej sesji można skopiować lub pobrać jako TXT;
uruchomienie nowej sesji zastępuje wcześniejszy tekst. Przycisk **Zatrzymaj**
zamyka mikrofon i kończy przetwarzanie zebranych porcji. Wyjście z zakładki lub
zamknięcie karty przerywa nasłuch i porzuca jeszcze nieprzetworzone porcje.
Nagrywanie i napisy nie korzystają jednocześnie z mikrofonu, a podczas sesji
GPU inne zadania GPU oczekują. Jeśli rozpoznawanie nie nadąża, aplikacja
ogranicza kolejkę dźwięku i ostrzega o pominiętych sekundach. Granica porcji
może przeciąć słowo; sesja napisów nie zastępuje dokładnej transkrypcji pliku.

## Rozróżnianie rozmówców

W formularzu **Transkrypcja** lub **Napisy na żywo** włącz przełącznik
**Rozróżniaj rozmówców**. Silnik oznaczy głosy jako `Rozmówca 1`, `Rozmówca 2`
itd. Numery są przypisywane od nowa dla każdego nagrania lub sesji i nie
identyfikują konkretnych osób. W podglądzie i eksporcie TXT, SRT, VTT, JSON i CSV
zachowywane są etykiety przypisane fragmentom; w edytorze można je poprawić.

Przy pierwszym użyciu aplikacja pobiera dwa modele o łącznym rozmiarze około
46 MB do podkatalogu `speaker-diarization` w wybranym katalogu modeli i
sprawdza ich sumy SHA-256. Kolejne użycia działają offline. Dźwięk pozostaje na
komputerze. Analiza głosów działa na CPU, także gdy Whisper używa GPU, i może
zwiększyć czas przetwarzania. Przy krótkich wypowiedziach, podobnych głosach,
hałasie lub jednoczesnej mowie etykiety mogą być niepewne albo błędne. Każda
porcja jest analizowana osobno, a głosy dopasowywane między porcjami; wyniki
warto sprawdzić w edytorze. Jedna sesja obsługuje do 20 etykiet rozmówców.
Funkcja oznacza tekst, nie rozdziela pliku audio.

Wykorzystane modele to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
w [wersji ONNX projektu sherpa-onnx](https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0)
oraz [3D-Speaker ERes2Net](https://huggingface.co/csukuangfj/speaker-embedding-models/blob/main/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx).
Silnik diarizacji zapewnia [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

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
  Anulowanie inferencji następuje przy najbliższej granicy segmentu lub porcji.
  Anulowanie zadania w kolejce czeka na zakończenie pobierania modelu; można je
  przerwać bezpośrednio w zakładce **Modele AI**.
- `RTF` to czas zadania podzielony przez długość nagrania; czas obejmuje również
  ładowanie modelu i przygotowanie. Mniejsza wartość oznacza szybsze działanie.
- Nagrywanie wymaga działającego wejścia PortAudio w systemie. Dla nowo
  zaimportowanego wideo edytor odtwarza zapisaną ścieżkę audio; odtwarzanie
  zachowanych wcześniej plików wideo zależy od kodeków przeglądarki.

## Diagnostyka i utrzymanie

**Diagnostyka → Uruchom test CUDA** ładuje model `tiny` i wykonuje prawdziwą
inferencję na karcie wybranej w diagnostyce, w osobnym procesie. Pokaże też
błędy brakujących DLL oraz listę kart widocznych dla CUDA. W konsoli:

```powershell
.\.venv\Scripts\python.exe -m app.core.gpu_manager --test
.\.venv\Scripts\python.exe -m app.core.gpu_manager --test --gpu-id cuda:1
powershell -ExecutionPolicy Bypass -File .\repair.ps1
powershell -ExecutionPolicy Bypass -File .\repair.ps1 -Full
powershell -ExecutionPolicy Bypass -File .\update.ps1
```

Opcja `--gpu-id cuda:1` wybiera drugi aktualnie widoczny indeks CUDA; można też
podać UUID z raportu diagnostycznego. Instalator i skrypty naprawy oraz
aktualizacji przyjmują parametr `-GpuId cuda:1`. Jeśli test pierwszej karty
blokuje pierwszą instalację, można użyć `install.ps1 -SkipGpuTest`, uruchomić
aplikację, wybrać inną kartę i wykonać test w diagnostyce.

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
