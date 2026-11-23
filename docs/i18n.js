// French translations of the static page; English is the HTML itself.
const I18N_FR = {
  title: "Analyse de sentiment &amp; classification de texte — avis clients aériens",
  h1: "Analyse de sentiment &amp; classification de texte",
  tagline: "Classer les retours clients par sentiment et par thème de plainte pour faire ressortir ce qui compte pour les utilisateurs.",
  btnRepo: "Dépôt GitHub", btnDemo: "Démo en direct", btnViz: "Visualisations d'évaluation",
  kRows: "tweets après nettoyage", kBest: "meilleur macro-F1 sentiment (test)",
  kBase: "baseline TF-IDF + régression logistique", kTopic: "macro-F1 thème de plainte (9 classes)",
  h01: "Problème &amp; contexte métier",
  p01: `Les compagnies aériennes reçoivent des milliers de tweets par jour ; la plupart ne sont jamais lus par un humain. Des étiquettes automatiques de <b>sentiment</b>
  (négatif / neutre / positif) transforment ce flux en un signal mesurable, et des étiquettes de <b>thème</b> sur les plaintes (vol en retard, bagage perdu,
  service client, …) indiquent aux équipes opérationnelles <i>quoi</i> corriger. Utilisateurs du résultat : les équipes de relation client sur les réseaux sociaux
  (traiter d'abord les tweets négatifs) et les managers ops/CX (tableaux de bord hebdomadaires des causes de plainte par compagnie).`,
  h02: "Jeu de données &amp; sources",
  p02: `<b>Twitter US Airline Sentiment</b> (CrowdFlower / Figure Eight, fév. 2015 ; via Hugging Face <code>osanseviero/twitter-airline-sentiment</code>) :
  <span data-k="raw"></span> tweets adressés à 6 compagnies américaines, annotés à la main par des contributeurs.`,
  ul02: `<li><b>Tâche A — sentiment</b> : 3 classes <code>negative</code> / <code>neutral</code> / <code>positive</code>.</li>
    <li><b>Tâche B — thème de plainte</b> : pour les tweets négatifs, 10 étiquettes « negative reason » ; <code>Can't Tell</code> est retirée, il reste 9 classes.</li>`,
  h03: "Nettoyage &amp; prétraitement",
  ul03: `<li>Décodage des entités HTML, passage en minuscules, normalisation des apostrophes typographiques.</li>
    <li>Suppression des URL et des <b>@mentions</b> — elles nomment la compagnie, ce qui ferait fuiter la compagnie dans le modèle.</li>
    <li>Contractions qui préservent la négation (<code>can't → can not</code>, <code>n't → not</code>) ; pas de suppression des mots vides, car « not » inverse le sentiment.</li>
    <li>Tokenisation par expression régulière qui garde les mots, <code>!</code>/<code>?</code> et les emoji comme tokens.</li>
    <li>Suppression des doublons exacts après nettoyage (retweets/copies) : <span data-k="dups"></span> lignes retirées, aucun doublon ne peut donc être à la fois en entraînement et en test.</li>
    <li>Détection de langue (<code>langdetect</code>) : <span data-k="flagged"></span> tweets signalés comme non anglais — à l'inspection c'était de l'argot/des emoji en anglais
      (« sounds like fun ! », « i ❤️ flying »), le signal est donc rapporté mais <b>pas</b> utilisé comme filtre.</li>`,
  p03: "Le même nettoyage et la même tokenisation sont réimplémentés en JavaScript pour que la démo ci-dessous exécute la baseline entraînée dans votre navigateur.",
  h04: "Analyse exploratoire des données",
  p04: `Les classes sont déséquilibrées (<span data-k="balance"></span>) : l'accuracy seule flatterait un modèle qui prédit trop « négatif ».
  Les tweets négatifs sont plus longs (médiane <span data-k="lenNeg"></span> tokens contre <span data-k="lenPos"></span> pour les positifs) — les gens expliquent ce qui s'est mal passé.`,
  cBalance: "Répartition des classes", cLength: "Distribution des longueurs (tokens)",
  cTopics: "Thèmes de plainte (les 10 étiquettes, avant retrait de « Can't Tell »)", cNgrams: "Bigrammes les plus fréquents par classe",
  capNgrams: `Mots vides retirés pour l'affichage uniquement. Négatif : « customer service », « cancelled flightled » (artefact de la source, où « Cancelled Flight » a été inséré automatiquement), « been hold » (… on hold). Positif : « great job », « thanks much », « 🙏 🙏 » — et encore « customer service » : le même bigramme apparaît dans les éloges et dans les plaintes.`,
  h05: "Ingénierie des variables",
  ul05: `<li><b>Creuse</b> : TF-IDF sur les uni- et bi-grammes de mots (min_df=2, tf sous-linéaire, 20k variables) — les bigrammes capturent « not good », « on hold », « thank you ».</li>
    <li><b>Dense</b> : TF-IDF → TruncatedSVD (300 composantes) pour le gradient boosting, qui gère mal 20k colonnes creuses.</li>
    <li><b>Contextuelle</b> : embeddings de phrases <code>all-MiniLM-L6-v2</code> figés (384 dimensions), et fine-tuning complet d'un transformer (DistilRoBERTa) avec son propre tokeniseur en sous-mots.</li>`,
  h06: "Sélection de modèle", p06: "Baselines contre transformers, sur le même découpage :",
  h07: "Entraînement &amp; validation",
  p07: `Découpage stratifié 70 / 15 / 15 entraînement / validation / test (graine 42). Les hyperparamètres (p. ex. le <code>C</code> de la régression logistique) et l'époque
  de fine-tuning sont choisis sur le <b>macro-F1 de validation</b> ; le jeu de test n'est utilisé qu'une fois. Poids équilibrés par classe dans chaque modèle pour que les classes minoritaires comptent.`,
  h08: "Métriques d'évaluation",
  p08: `Le <b>macro-F1</b> est la métrique principale (chaque classe pèse autant). Également rapportés : rappel par classe, matrices de confusion, ROC-AUC un-contre-tous,
  et intervalles de confiance bootstrap à 95 % (1 000 rééchantillonnages du jeu de test) sur le macro-F1.`,
  h09: "Résultats", hRecall: "Rappel par classe (sentiment)", hTopicTbl: "Thème de plainte (tweets négatifs uniquement)",
  h10: "Limites",
  ul10: `<li><b>Sarcasme &amp; tweets mitigés</b> : « great, another 3 hour delay. love it. » est classé positif par la baseline sac de mots.</li>
    <li><b>Bruit d'annotation</b> : annotations participatives ; la plupart des erreurs les plus confiantes ci-dessous sont de courts « thanks! » annotés <i>neutre</i>, ou des plaintes détaillées annotées neutre.</li>
    <li><b>Changement de domaine</b> : une semaine de tweets de 2015 sur des compagnies américaines ; le vocabulaire (et Twitter lui-même) a changé.</li>
    <li><b>Texte multilingue</b> : le corpus est en anglais ; le modèle TF-IDF ne couvre aucune autre langue et la détection de langue est peu fiable sur des textes courts.</li>
    <li>Les thèmes n'existent que pour les tweets négatifs, et plusieurs thèmes ont moins de 100 exemples.</li>`,
  hErrors: "Erreurs les plus confiantes (baseline)", h11: "Leçons apprises",
  hDemo: `Démo en direct <span class="badge">tourne dans votre navigateur</span>`,
  pDemo: "Les modèles TF-IDF + régression logistique entraînés (sentiment + thème), exportés en JSON — sans serveur. Les contributions des mots sont exactes pour un modèle linéaire. Le modèle a été entraîné sur des tweets en anglais : écrivez en anglais.",
  dSent: "Sentiment", dTopic: `Thème de plainte <span class="muted" style="font-weight:400;font-size:13px">(si négatif)</span>`,
  dWhy: "Pourquoi ? Contribution des mots à la classe prédite (log-odds)",
  hViz: "Visualisations d'évaluation",
  pViz: `Tous les graphiques utilisent le vrai jeu de test de ce projet (<span data-k="ntest"></span> tweets).`,
  vCompare: "Comparaison des performances des modèles",
  capCompare: "Métriques en moyenne macro ; ROC-AUC en un-contre-tous. Baseline interprétable contre modèles non linéaires et transformers.",
  vCM: "Matrice de confusion — 3 classes", capCM: "Lignes = réel, colonnes = prédit. La plupart des erreurs sont neutre ↔ négatif.",
  vBin: "Alerte « à traiter » : négatif contre le reste", thr: "Seuil de décision",
  capBin: "Déplacez le seuil pour arbitrer entre précision (moins de fausses alertes) et rappel (moins de plaintes manquées).",
  vROC: "Courbe ROC — négatif contre le reste", capROC: "AUC 0,5 = hasard, 1,0 = parfait.",
  vPR: "Courbe précision–rappel — négatif contre le reste", capPR: "Plus informative que la ROC quand la classe positive est rare.",
  vImp: "Importance des variables (random forest)", capImp: "Baisse moyenne d'impureté de chaque terme TF-IDF sur les 400 arbres.",
  vCoef: "Termes principaux par classe (régression logistique)",
  vLoss: "Perte d'entraînement vs validation",
  capLoss: "Des courbes qui divergent signalent du sur-apprentissage ; l'arrêt précoce / le choix de la meilleure époque utilise la courbe de validation.",
  vExp: "Explication de type SHAP pour un tweet de test",
  capExp: "Contribution additive de chaque n-gramme (tf-idf × coefficient) au log-odds de la classe prédite.",
  footer: `Données : Twitter US Airline Sentiment (CrowdFlower, CC BY-NC-SA 4.0). Code : <a href="https://github.com/SYNDOUKONE/sentiment-text-classification">github.com/SYNDOUKONE/sentiment-text-classification</a>.`,
};
const CLASS_FR = {negative: "négatif", neutral: "neutre", positive: "positif"};
const TOPIC_FR = {"Bad Flight": "Mauvais vol", "Can't Tell": "Indéterminé", "Cancelled Flight": "Vol annulé", "Customer Service Issue": "Service client",
  "Damaged Luggage": "Bagage endommagé", "Flight Attendant Complaints": "Personnel de bord", "Flight Booking Problems": "Problème de réservation",
  "Late Flight": "Vol en retard", "Lost Luggage": "Bagage perdu", "longlines": "Longues files d'attente"};
