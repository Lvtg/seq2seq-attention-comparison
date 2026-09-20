# Error Analysis Case Samples

## Case 1 | test #848 | 1-10 | rare word; fluent but incomplete

- Source: a gymnast is being judged at en event .
- Reference: eine turnerin wird bei einer veranstaltung beurteilt .
- LSTM no attention greedy: eine zeigt bei einer veranstaltung .
- LSTM attention selected: eine turnerin macht .
- Gated attention selected: ein turner wartet auf .
- Small Transformer selected: eine turnerin wird bei einer bei einem .
- Note: Best by token F1: transformer_selected; worst: gated_attention_selected; selected for high model disagreement.

## Case 2 | test #473 | 1-10 | action error; correct/paraphrase

- Source: a man fixing a little girl ' s bicycle .
- Reference: ein mann repariert das fahrrad eines kleinen mädchens .
- LSTM no attention greedy: ein mann schneidet einem kleinen mädchen das essen .
- LSTM attention selected: ein mann fährt ein kleines mädchen das fahrrad .
- Gated attention selected: ein mann bedient ein kleines mädchen .
- Small Transformer selected: ein mann repariert das fahrrad eines kleinen mädchens .
- Note: Best by token F1: transformer_selected; worst: gated_attention_selected; selected for high model disagreement.

## Case 3 | test #5 | 1-10 | action error; correct/paraphrase

- Source: people are fixing the roof of a house .
- Reference: leute reparieren das dach eines hauses .
- LSTM no attention greedy: leute machen sich von einem gebäude auf dem wasser .
- LSTM attention selected: leute ziehen das dach eines hauses .
- Gated attention selected: leute führen das dach eines hauses .
- Small Transformer selected: mehrere personen reparieren das dach eines hauses .
- Note: Best by token F1: lstm_attention_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 4 | test #115 | 1-10 | correct/paraphrase

- Source: two indian men participating in a ceremony .
- Reference: zwei indische männer nehmen an einer zeremonie teil .
- LSTM no attention greedy: zwei männer sich in einem laden .
- LSTM attention selected: zwei indische männer nehmen an einem laden teil .
- Gated attention selected: zwei indische männer nehmen an einem teil .
- Small Transformer selected: zwei indische männer nehmen an einer zeremonie teil .
- Note: Best by token F1: transformer_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 5 | test #949 | 1-10 | action error; correct/paraphrase

- Source: people playing a game in the pool .
- Reference: leute spielen ein spiel im pool .
- LSTM no attention greedy: leute spielen in einem spiel im spiel .
- LSTM attention selected: menschen spielen ein spiel im pool .
- Gated attention selected: leute spielen in einem spiel im sand .
- Small Transformer selected: menschen spielen in einem schwimmbecken .
- Note: Best by token F1: lstm_attention_selected; worst: transformer_selected; selected for high model disagreement.

## Case 6 | test #192 | 11-20 | correct/paraphrase

- Source: a man next to a bicycle is playing a pan flute .
- Reference: ein mann spielt neben einem fahrrad panflöte .
- LSTM no attention greedy: ein mann , der einen - wagen trägt , macht eine pause .
- LSTM attention selected: ein mann neben einem fahrrad spielt ein instrument .
- Gated attention selected: ein mann neben einem fahrrad spielt flöte .
- Small Transformer selected: ein mann spielt neben einem ungewöhnlichen fahrrad .
- Note: Best by token F1: gated_attention_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 7 | test #44 | 11-20 | correct/paraphrase

- Source: a kid swings with his feet up in the air in a forest .
- Reference: ein junge schaukelt mit den füßen in der luft in einem wald .
- LSTM no attention greedy: ein kind steckt seine füße in einem park mit seinem maul .
- LSTM attention selected: ein kind schwimmt mit seinem kopf in einem wald in die luft .
- Gated attention selected: ein kind steckt mit seinem sohn in einem wald in die luft .
- Small Transformer selected: ein kind schaukelt in einem wald mit den füßen in der luft .
- Note: Best by token F1: transformer_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 8 | test #584 | 11-20 | rare word

- Source: military families are marching through new york on a rainy day .
- Reference: militärangehörige marschieren an einem regnerischen tag durch new york .
- LSTM no attention greedy: gehen in der nacht auf einer durch einen überfüllten freiluftmarkt .
- LSTM attention selected: hier sind durch aufgehängt auf einem geschäftigen markt .
- Gated attention selected: sind an einem kalten tag durch san francisco .
- Small Transformer selected: soldaten laufen an einem regnerischen tag durch einen new york .
- Note: Best by token F1: transformer_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 9 | test #654 | 11-20 | repetition

- Source: music being played by several individuals while a happy crowd sits and listens .
- Reference: mehrere personen spielen musik während eine gut gelaunte menge sitzt und zuhört .
- LSTM no attention greedy: es sind draußen versammelt , umgeben von personen und sitzen umgeben von einer bank .
- LSTM attention selected: eine wird von mehreren , während ein paar erwachsene herumstehen und .
- Gated attention selected: eine menschenmenge wird von mehreren männern gezeigt , die von einer getragen .
- Small Transformer selected: eine menschenmenge sitzt und hört musik , während mehrere personen .
- Note: Best by token F1: transformer_selected; worst: gated_attention_selected; selected for high model disagreement.

## Case 10 | test #141 | 1-10 | rare word; repetition; correct/paraphrase

- Source: a fox terrier leaps after a ball .
- Reference: ein foxterrier springt nach einem ball .
- LSTM no attention greedy: ein fängt einen ball beim fangen .
- LSTM attention selected: ein springt mit einem ball .
- Gated attention selected: ein springt nach einem ball .
- Small Transformer selected: ein springt nach einem ball nach einem ball .
- Note: Best by token F1: gated_attention_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 11 | test #216 | 1-10 | rare word

- Source: people chilling in at forest next to canoes .
- Reference: leute entspannen in einem wald neben kanus .
- LSTM no attention greedy: leute in gehen in der nacht .
- LSTM attention selected: menschen in einem wald auf dem wasser .
- Gated attention selected: menschen sich in einem studio an bord verschiedener boote .
- Small Transformer selected: menschen in kanus neben einem wald .
- Note: Best by token F1: transformer_selected; worst: gated_attention_selected; selected for high model disagreement.

## Case 12 | test #153 | 1-10 | action error; correct/paraphrase

- Source: a woman is holding a small white statue .
- Reference: eine frau hält eine kleine weiße statue .
- LSTM no attention greedy: eine frau hält einen weißen langhaarigen hund .
- LSTM attention selected: eine frau hält ein kleines weißes rohr .
- Gated attention selected: eine frau hält ein kleines weißes menü .
- Small Transformer selected: eine frau hält eine kleine weiße statue .
- Note: Best by token F1: transformer_selected; worst: lstm_no_attention_greedy; selected for high model disagreement.

## Case 13 | test #874 | 31+ | action error; modifier loss; repetition; long sentence compression; fluent but incomplete

- Source: a man wearing a gray shirt , blue jeans and a neon green safety vest is standing on a railroad track with a white truck and a white building in the background .
- Reference: ein mann mit einem grauen shirt , blauen jeans und neongrüner sicherheitsweste steht mit einem weißen lieferwagen und einem weißen gebäude im hintergrund auf einer bahnstrecke .
- LSTM no attention greedy: ein mann in einem grauen oberteil und mit einem blauen oberteil , der in einem grauen gebäude mit einem grauen gebäude auf einem gebäude geparkt ist , fährt ein gelbes gebäude .
- LSTM attention selected: ein mann in einem grauen hemd , blauer jeans und einem grünen steht auf einem gerüst und hat einen weißen lkw mit einem weißen lkw und einem weißen gebäude im hintergrund .
- Gated attention selected: ein mann in einem grauen t - shirt und jeans fährt ein auf einer strecke mit einem weißen helm und einem weißen gebäude im hintergrund .
- Small Transformer selected: ein mann in einer weißen weste und einer blauen weste steht auf einer mit einem grünen lkw im hintergrund .
- Note: Best by token F1: gated_attention_selected; worst: lstm_no_attention_greedy; selected for long source sentence.

## Case 14 | test #960 | 31+ | action error; modifier loss; repetition; long sentence compression

- Source: a man in a black jacket and checkered hat wearing black and white striped pants plays an electric guitar on a stage with a singer and another guitar player in the background .
- Reference: ein mann mit kariertem hut in einer schwarzen jacke und einer schwarz - weiß gestreiften hose spielt auf einer bühne mit einem sänger und einem weiteren gitarristen im hintergrund auf einer e - gitarre .
- LSTM no attention greedy: ein mann in einer schwarzen jacke und mit einem weißen hut , der andere in einem schwarz - weiß gestreiften oberteil spielt , während ein kleines mädchen auf einem instrument spielt .
- LSTM attention selected: ein mann in einer schwarzen jacke und mit einem braunen hut , schwarzen hut und einem gestreiften hut spielt auf einer bühne mit einem stadtbild im hintergrund gitarre .
- Gated attention selected: ein mann in einer schwarzen jacke und mit hut und schwarzer kleidung spielt auf einer bühne gitarre und ein anderer mann spielt gitarre und ein anderer singt im hintergrund .
- Small Transformer selected: ein sänger in einer schwarzen jacke und einem weißen hut spielt auf der bühne mit einem anderen mann mit einem gestreiften hut und einer e - gitarre und einer e - gitarre .
- Note: Best by token F1: lstm_attention_selected; worst: lstm_no_attention_greedy; selected for long source sentence.

## Case 15 | test #358 | 21-30 | action error; repetition; long sentence compression; fluent but incomplete

- Source: two males seem to be conversing while standing in front of a truck ' s back , and behind a metal item , while four people stand around them .
- Reference: zwei männer stehen vor dem heck eines lasters und hinter einem metallgegenstand und unterhalten sich anscheinend während vier weitere personen um sie herum stehen .
- LSTM no attention greedy: zwei männer stehen vor einem stand , während zwei männer und eine andere person stehen , um ihn herum zu .
- LSTM attention selected: zwei männer scheinen sich vor einer reihe von männern und hinter ihnen stehen neben einem lkw und hinter ihnen stehen herum .
- Gated attention selected: zwei männliche security - stehen an einem lkw und werden von zwei personen hinter ihnen stehen , während andere menschen daneben zugucken .
- Small Transformer selected: vier personen stehen vor einem lkw , während zwei personen hinter ihnen steht und unterhalten .
- Note: Best by token F1: transformer_selected; worst: gated_attention_selected; selected for long source sentence.

## Case 16 | test #8 | 21-30 | action error; modifier loss; repetition; long sentence compression

- Source: a boy in a red uniform is attempting to avoid getting out at home plate , while the catcher in the blue uniform is attempting to catch him .
- Reference: ein junge in einem roten trikot versucht , die home base zu erreichen , während der catcher im blauen trikot versucht , ihn zu fangen .
- LSTM no attention greedy: ein junge in einem roten trikot versucht , den ball zu fangen , während er sich in der luft schwebt , während der fänger im hintergrund zusieht .
- LSTM attention selected: ein junge in einem roten trikot versucht , den ball zu treffen , während der schiedsrichter im blauen trikot abzuwehren .
- Gated attention selected: ein junge in einem roten trikot versucht den ball zu fangen , während der batter in das blau - spieler versucht , den wurf zu blocken .
- Small Transformer selected: ein junge in einem roten trikot versucht , den ball zu fangen , während ein rot - spieler in einem .
- Note: Best by token F1: lstm_attention_selected; worst: transformer_selected; selected for long source sentence.

## Case 17 | test #882 | 21-30 | modifier loss; repetition; long sentence compression; fluent but incomplete

- Source: two girls , one older and in black and one younger and in white , are performing the same ballet move in front of decorations made from balloons .
- Reference: zwei mädchen , eines älter und in schwarz , das andere jünger und in weiß , machen dieselbe tanzbewegung vor einer dekoration aus ballons .
- LSTM no attention greedy: zwei mädchen und ein junge – beide in schwarz - weißer kleidung – beteiligen sich vor einem auf dem tisch und warten auf den anderen .
- LSTM attention selected: zwei mädchen , eine von ihnen und eine frau , beide in weißen und lilafarbener kleidung , posieren für die vor der von 50 % .
- Gated attention selected: zwei mädchen , einer davon und ein schwarzer , befinden und ein anderer mann in weißer kleidung posieren vor einer reihe von ihnen und springen in die luft .
- Small Transformer selected: zwei ältere mädchen , eins in schwarz und ein jüngerer , führen einen tanz vor dem gleichen tanz auf .
- Note: Best by token F1: lstm_no_attention_greedy; worst: lstm_attention_selected; selected for long source sentence.

## Case 18 | test #828 | 21-30 | action error; repetition; long sentence compression; fluent but incomplete

- Source: three children in football uniforms of two different teams are playing football on a football field , while another player and an adult stand in the background .
- Reference: drei kinder in footballtrikots zweier verschiedener mannschaften spielen football auf einem footballplatz während ein weiterer spieler und ein erwachsener im hintergrund stehen .
- LSTM no attention greedy: drei kinder in schwarz spielen football , während ein spieler in einem trikot rennt , während ein spieler in einem blauen trikot zusieht .
- LSTM attention selected: drei kinder in trikots springen vor zwei gegnerischen mannschaften auf einem fußballfeld , während ein anderer trainer und ein schiedsrichter im hintergrund zuschaut .
- Gated attention selected: drei kinder in einem haus mit zwei instrumenten spielen fußball auf einem footballfeld , während ein anderer mann im hintergrund steht und hinter ihnen steht .
- Small Transformer selected: drei football - spieler spielen fußball , während ein anderer jugendlicher im hintergrund zuschaut .
- Note: Best by token F1: lstm_attention_selected; worst: gated_attention_selected; selected for long source sentence.

## Case 19 | test #316 | 11-20 | rare word; fluent but incomplete

- Source: the dali lama during a reception where participants have brought carnations , parasols and prayer flags .
- Reference: der dalai lama bei einem empfang , zu dem besucher nelken , schirme und gebetsfahnen bringen .
- LSTM no attention greedy: die und .
- LSTM attention selected: die macht einen , wobei sie und warten .
- Gated attention selected: die des wird eine , meine und meine .
- Small Transformer selected: ein king moment , während der und fahnen .
- Note: Best by token F1: transformer_selected; worst: lstm_attention_selected; selected for rare source tokens.

## Case 20 | test #532 | 21-30 | rare word; modifier loss; repetition; long sentence compression; fluent but incomplete

- Source: a solitary woman uses a large primitive mortar and pestle to crush plant material as she stands in a stubble field .
- Reference: eine einzelne frau verwendet einen großen einfachen stößel und mörser , um pflanzenmaterial zu zerdrücken , während sie auf einem stoppelfeld steht .
- LSTM no attention greedy: eine frau in einem , das ein großes trägt , hält ein großes und steht brusttief unter dem .
- LSTM attention selected: eine orientalische frau verwendet einen großen und , während sie in einem feld ein feld springt .
- Gated attention selected: eine orientalische frau jongliert mit aller kraft einen großen und abzuwehren , um ein gewicht zu .
- Small Transformer selected: eine frau steht auf einem feld und , während sie sich .
- Note: Best by token F1: transformer_selected; worst: lstm_no_attention_greedy; selected for rare source tokens.

## Case 21 | test #237 | 11-20 | rare word; action error; fluent but incomplete

- Source: a man , wearing revolutionary period clothes , is ringing a bell .
- Reference: ein mann mit kleidung aus dem unabhängigkeitskrieg schlägt eine glocke .
- LSTM no attention greedy: ein mann mit trägt einen blumenstrauß .
- LSTM attention selected: ein mann in kleidung eine .
- Gated attention selected: ein mann in kleidung eine .
- Small Transformer selected: ein mann in kleidung trägt ein - - .
- Note: Best by token F1: lstm_attention_selected; worst: lstm_no_attention_greedy; selected for rare source tokens.

## Case 22 | test #733 | 11-20 | rare word; fluent but incomplete

- Source: an oriental traveler awaits his turn at the currency exchange .
- Reference: ein orientalischer reisender wartet am wechselschalter bis er dran ist .
- LSTM no attention greedy: ein - nimmt an einem teil .
- LSTM attention selected: ein mit dem bei der .
- Gated attention selected: ein nackter cowboy - überprüft seine am .
- Small Transformer selected: ein orientalischer regelt den , dass sein tag .
- Note: Best by token F1: gated_attention_selected; worst: lstm_attention_selected; selected for rare source tokens.

## Case 23 | test #7 | 11-20 | correct/paraphrase

- Source: a group of people standing in front of an igloo .
- Reference: eine gruppe von menschen steht vor einem iglu .
- LSTM no attention greedy: eine gruppe leute steht vor einem haus .
- LSTM attention selected: eine gruppe von menschen steht vor einem .
- Gated attention selected: eine gruppe von menschen steht vor einer .
- Small Transformer selected: eine gruppe von menschen steht vor einem iglu .
- Note: Best by token F1: transformer_selected; worst: lstm_no_attention_greedy; selected for strong/correct example.

## Case 24 | test #9 | 1-10 | correct/paraphrase

- Source: a guy works on a building .
- Reference: ein typ arbeitet an einem gebäude .
- LSTM no attention greedy: ein mann arbeitet an einem gebäude .
- LSTM attention selected: ein typ arbeitet an einem gebäude .
- Gated attention selected: ein typ arbeitet an einem gebäude .
- Small Transformer selected: ein mann arbeitet an einem gebäude .
- Note: Best by token F1: lstm_attention_selected; worst: lstm_no_attention_greedy; selected for strong/correct example.

