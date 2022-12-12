# Data project with Quinten: Posology extraction from patient reports – Named Entity Recognition 🏥 

## Introduction 💊

Drug posology concerns all the conditions of  administration of a treatment to a patient. It is  frequently expressed in patient admission reports in  a standardized way.
The detection of the different attributes related to  posology is critical to perform statistics and patient monitoring on a long term perspective.

In our case, the posology attributes are the following:

- Drug: Molecule or medicine commercial name of the medicine provided during a treatment to a patient
- Treatment: Any type of treatment that is not a molecule or a medicine
- Dosage: Dosage of the medicine provided (including the unit)
- Frequency: Frequency of the medicine delivery
- Form: Type of the medicine (tablet, drop, bag ...)
- Route: Way under which the treatment is provided (intravenous, oral, ...)
- Duration: Period of time the treatment is provided

We perform a text classification from 567 french raw admission/medical discharge reports.


## Data augmentation ⤴

### Synonyms replacement (not used)

Synonyms replacement is described in the notebook located at [code/data%20augmentation/extract_forms_routes.ipynb](code/data%20augmentation/extract_forms_routes.ipynb). Due to numerous inconsistencies in the synonyms librairies that do not rely on words' context and environment, we were unable to use this data augmentation framework.

### Backtranslation 🇫🇷🇬🇧🇫🇷

We performed backtranslation on our dataset, mainly with english intermediary language.

### Labels shuffling 🗳

In order to perform some shuffling within each label, it is necessary to first collect some external data. We used databases provided by [ANSM (Agence nationale de sécurité du médicament et de produits de santé)](https://base-donnees-publique.medicaments.gouv.fr/telechargement.php) with the aim of extracting:


*   drug names from the generic drugs database
*   routes and forms from the marketed drugs database

The extraction process is described in the notebook located at [code/data%20augmentation/extract_forms_routes.ipynb](code/data%20augmentation/extract_forms_routes.ipynb), which generates text files as output containing generic lists for each label. These text files can then be processed to feed the final model with suffled entries.

## Tokenization 🪙

We used the platform Doccano for manual text labellisation, and CamemBERT for tokenization, which is a state-of-the-art language model for French based on the RoBERTa model. CamemBERT can only deal with numerical class values, so we had to adapt the initial output of Doccano.

The scoring of the final model was made with Kaggle, so we had to convert our output to a specific submission model.

The functions for tokenization and labels management are stored in [main/code/tokens_labels_management](main/code/tokens_labels_management)

## Running the NER CamemBERT-based model

The final NER model is located at [code/model/ner_model.ipynb](code/model/ner_model.ipynb).

## Dealing with RegEx ✎

### RegEx naive model

To perform some naive predictions that only involve RegEx and if statements, open the notebook [code/model/naive_regex.ipynb](code/model/naive_regex.ipynb) and launch all the cells.

The CSV file finally generated contains the predictions performed by the model and can be evaluated on the Kaggle submission page.

### RegEx layer on NER CamemBERT model 🧀

Predictions performed by the NER CamemBERT model can be improved with an additionnal layer involving RegEx and if statements based on the naive version described above.

The steps are defined in the notebook standing at [code/model/regex_layer.ipynb](code/model/regex_layer.ipynb)

The data we used is not available in this repository 🔒

@François Schmerber, Pierre de Boisredon, Thomas Kessous, Sylvain Delgendre, Clément Girault
