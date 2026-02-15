# 🤖 Tehisintellekti rakendamise projektiplaani mall (CRISP-DM)

<br>
<br>


## 🔴 1. Äritegevuse mõistmine
*Fookus: mis on probleem ja milline on hea tulemus?*


### 🔴 1.1 Kasutaja kirjeldus ja eesmärgid
Kellel on probleem ja miks see lahendamist vajab? Mis on lahenduse oodatud kasu? Milline on hetkel eksisteeriv lahendus?

> Õppejõud ja tudengid vajavad kiiret ja mugavat lahendust ÕIS-i õppeainete otsinguks. Tartu ülikooli õppeinfo süsteemis on tuhandeid õppeaineid ning endale sobivate õppeainete leidmine võib olla üsna keeruline ning aeganõudev - ükshaaval ainete läbikäimine ei ole realistlik ning praegused väga konkreetsete otsingusõnadega filtrid ei ole piisavalt paindlikud. Lisaks pole võimalik näha eelmiste semestrite aineid. Kuna üks Tartu Ülikooli plussidest on just võimalus võtta väga erinevaid õppeaineid ning panna endale kokku just sulle sobiv kava, on eriti vaba- ja valikainete parem otsimine väga tore võimalus tudengite jaoks leida just endale sobivad õppeained. Kasu ongi see, et saame leida paremaid sobivusi ainete ning tudengite vahel ning teha protsessi lõbusamaks ning ehk ka just integreerida erinevate valdkondade kontakte.

### 🔴 1.2 Edukuse mõõdikud
Kuidas mõõdame rakenduse edukust? Mida peab rakendus teha suutma?

> Vastuse ajakulu ja kvaliteedi suhte järgi. Rakendus peab suutma andma relevantset infot antud sisendi korral. Võetakse rohkem mitte oma õppekava aineid. Saame koguda kasutaja tagasisidet. Struktureeritud testimine teststsenaariumitega. Rakendus ei leia ebasobivaid õppeaineid.

### 🔴 1.3 Ressursid ja piirangud
Millised on ressursipiirangud (nt aeg, eelarve, tööjõud, arvutusvõimsus)? Millised on tehnilised ja juriidilised piirangud (GDPR, turvanõuded, platvorm)? Millised on piirangud tasuliste tehisintellekti mudelite kasutamisele?

> Aega on neli nädalat ning eelarve on minimaalne. Rakendus võiks töötada avalikel ÕIS2 andmetel ning olla veebipõhine ning vabalt kättesaadav (praeguse aine raames jääb see jooksma lokaalselt). Rakendus peaks kasutama kas vabavaralisi tehisintellekti mudeleid või kui rakenduse edukaks toimimiseks on vaja tasulisi mudeleid, siis tuleb kindlasti vaadata, et kasutamisel oleks piirang vastavalt ressursi olemasolule. Antud projekti raames on meil arenduseks aega 1 kuu ning umbes 50 EURi 20 inimese peale tasuliste mudelite kasutamiseks. Rakendus ei tohi anda kasutajale ebasobivaid ja õppeainete otsinguga mitteseotud vastuseid.

<br>
<br>


## 🟠 2. Andmete mõistmine
*Fookus: millised on meie andmed?*

### 🟠 2.1 Andmevajadus ja andmeallikad
Milliseid andmeid (ning kui palju) on lahenduse toimimiseks vaja? Kust andmed pärinevad ja kas on tagatud andmetele ligipääs?

> Andmed pärinevad kohalikust CSV failist "toorandmed.csv". ÕISi API. Viimase 2 aasta andmed, õppeastme/õppekava kuuluvuse informatsioon, kohapeal või veebis. Vajalikud oleksid ligikaudu 20 veergu, olenevalt ajalisest piirangust on mõistlik lisada ka väiksema prioriteediga veerud, mida on ligikaudselt 40.

### 🟠 2.2 Andmete kasutuspiirangud
Kas andmete kasutamine (sh ärilisel eesmärgil) on lubatud? Kas andmestik sisaldab tundlikku informatsiooni?

> Andmestik sisaldab telefoninumbreid. Andmete kasutamine sõltub Tartu Ülikooli API kasutustingumustest.

### 🟠 2.3 Andmete kvaliteet ja maht
Millises formaadis andmeid hoiustatakse? Mis on andmete maht ja andmestiku suurus? Kas andmete kvaliteet on piisav (struktureeritus, puhtus, andmete kogus) või on vaja märkimisväärset eeltööd?

> Andmeid hoiustatakse formaatides String, JSON, Boolean, Float, Integer. Andmestik sisaldab 3031 andmepunkti ja 223 tunnust. Hädavajalikud väljad on puhtad. 

### 🟠 2.4 Andmete kirjeldamise vajadus
Milliseid samme on vaja teha, et kirjeldada olemasolevaid andmeid ja nende kvaliteeti.

> On vaja valida õige veerg info leidmiseks, puhastada json väljad, panna kokku vabatekstilised kirjeldavad tunnused keelemudelile või RAG süsteemile analüüsiks. Vaja on üle vaadata puuduvate tunnuste hulk ning otsustada, mida nendega ette võtta. 

<br>
<br>


## 🟡 3. Andmete ettevalmistamine
Fookus: Toordokumentide viimine tehisintellekti jaoks sobivasse formaati.

### 🟡 3.1 Puhastamise strateegia
Milliseid samme on vaja teha andmete puhastamiseks ja standardiseerimiseks? Kui suur on ettevalmistusele kuluv aja- või rahaline ressurss?

> Andmed on vaja puhastada natukene sarnasel viisil nagu 2.4 andmete kirjelduses mainitud. Võimalik, et oleks vaja imputeerida puuduvaid andmeid või neid otsida mõnest teisest ÕIS2 APIst või järeldada muudest andmetest. Andmete puhastamisele võiks kuluda umbes 1 nädal.

### 🟡 3.2 Tehisintellektispetsiifiline ettevalmistus
Kuidas andmed tehisintellekti mudelile sobivaks tehakse (nt tükeldamine, vektoriseerimine, metaandmete lisamine)?

> Olenevalt erinevatest meetoditest saame anda tehisintellektile kirjelduse andmetest ning ligipääsu puhastatud andmetele, et neid vajadusel filtreerida jne. RAG süsteemi jaoks on vaja välja valida aineid kirjeldavad veerud ning teha iga aine jaoks üks kirjeldav tekst. Valitud andmed tuleb vektoresituse kujule viimise mudeliga teisendada vektoriteks. Selle abil saab RAG süsteem semantiliselt otsingu järgi valida otsingule vastavad ained.

<br>
<br>

## 🟢 4. Tehisintellekti rakendamine
Fookus: Tehisintellekti rakendamise süsteemi komponentide ja disaini kirjeldamine.

### 🟢 4.1 Komponentide valik ja koostöö
Millist tüüpi tehisintellekti komponente on vaja rakenduses kasutada? Kas on vaja ka komponente, mis ei sisalda tehisintellekti? Kas komponendid on eraldiseisvad või sõltuvad üksteisest (keerulisem agentsem disan)?

> ...

### 🟢 4.2 Tehisintellekti lahenduste valik
Milliseid mudeleid on plaanis kasutada? Kas kasutada valmis teenust (API) või arendada/majutada mudelid ise?

> ...

### 🟢 4.3 Kuidas hinnata rakenduse headust?
Kuidas rakenduse arenduse käigus hinnata rakenduse headust?

> ...

### 🟢 4.4 Rakenduse arendus
Milliste sammude abil on plaanis/on võimalik rakendust järk-järgult parandada (viibadisain, erinevte mudelite testimine jne)?

> ...


### 🟢 4.5 Riskijuhtimine
Kuidas maandatakse tehisintellektispetsiifilisi riske (hallutsinatsioonid, kallutatus, turvalisus)?

> ...

<br>
<br>

## 🔵 5. Tulemuste hindamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🔵 5.1 Vastavus eesmärkidele
Kuidas hinnata, kas rakendus vastab seatud eesmärkidele?

> ...

<br>
<br>

## 🟣 6. Juurutamine
Fookus: kuidas hinnata loodud lahenduse rakendatavust ettevõttes/probleemilahendusel?

### 🟣 6.1 Integratsioon
Kuidas ja millise liidese kaudu lõppkasutaja rakendust kasutab? Kuidas rakendus olemasolevasse töövoogu integreeritakse (juhul kui see on vajalik)?

> ...

### 🟣 6.2 Rakenduse elutsükkel ja hooldus
Kes vastutab süsteemi tööshoidmise ja jooksvate kulude eest? Kuidas toimub rakenduse uuendamine tulevikus?

> ...