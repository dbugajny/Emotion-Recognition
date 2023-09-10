### Opis plików:

**Folder src:**

base_transformer - odpowiada za podstawowe przekształcenia: wybranie odpowiednich kolumn, zmianę nazw oraz typów,
ekstrację ID badanej osoby z nazwy plików. Inne transoformery dziedziczą po klasie BaseTransformer.

annotations_transformer - zawiera parametry dla podstawowych transformaci dla plików typu annotations

bitalino_transformer - zawiera parametry dla podstawowych transformaci dla plików typu bitalino,
odpowiada za wyliczenie featurów do treningu

bitalino_transformer - zawiera parametry dla podstawowych transformaci dla plików typu fixations,
odpowiada za wyliczenie featurów do treningu

pupil_positions_transformer - zawiera parametry dla podstawowych transformaci dla plików typu pupil positions,
określenie występowania sakad, odpowiada za wyliczenie featurów do treningu,

raiting_transformer - zawiera parametry dla podstawowych transformaci dla plików typu raiting,
odpowiada za łączenie z plikiem key_pictures

model_training - zawiera funkcje odpowiadające za trening modelu, trening modelu opiera się na wykonaniu grid searchu
i wybraniu najlepszego modelu

utils - zawiera funkcję do obliczania statystyk dla podanej kolumny oraz funkcję umożliwiającą obliczenie sakad

**Folder scripts:**

constants - zawiera stałe w postaci ścieżek do plików i folderów oraz instancje transformerów

hyperparameters - zawiera modele wraz z hiperparametrami używane w grid searchu oraz kolumny, które są brane pod uwagę
w treningu

perform_base_cleaning - przygotowuje plik key pictures oraz dokonuje podstawowego czyszczenia plików:
wybiera odpowiednie kolumny, zmienia ich typy oraz nazwy

perform_pictures_merging - łączy odpowiednie pliki - raitings z key pictures oraz fixations, pupil positions i bitalino
z połączonym wcześniej plikiem z ocenami

perform_features_calculating - oblicza odpowiednie featury

perform_features_merging - łączy featury z różnich źródeł, tworzy zbiór treningowy oraz testowy

perform_model_training - przeprowadza grid search, wybierana najlepszy model z podanych możliwości
