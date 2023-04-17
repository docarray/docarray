


<a name=release-note-0-30-0></a>
## Release Note (`0.30.0`)

> Release time: 2023-04-17 16:35:12



🙇 We'd like to thank all contributors for this new release! In particular,
 samsja,  Charlotte Gerhaher,  Alex Cureton-Griffiths,  Nan Wang,  Anne Yang,  Johannes Messner,  Nikolas Pitsillos,  Kacper Łukawski,  Shukri,  Saba Sturua,  Aman Agarwal,  Aziz Belaweid,  Jackmin801,  AlaeddineAbdessalem,  Tanguy Abel,  Joan Fontanals,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```2ea0acd7```](https://github.com/jina-ai/docarray/commit/2ea0acd7dc94501a10b1f824a2e8c1138fb94298)] __-__ qdrant document index (#1321) (*Kacper Łukawski*)
 - [[```046c8deb```](https://github.com/jina-ai/docarray/commit/046c8debf57751389482ecef84799dc154d9a6f1)] __-__ weaviate document index V2! (#1367) (*Shukri*)
 - [[```bd41ce64```](https://github.com/jina-ai/docarray/commit/bd41ce64693233912e920b3008beabe61014ab0c)] __-__ elasticsearch(v8) document index  (#1251) (*Anne Yang*)
 - [[```95c1eca3```](https://github.com/jina-ai/docarray/commit/95c1eca3564d1bc59a66b43dc877067d6f8b315e)] __-__ read from remote csv file (#1362) (*Charlotte Gerhaher*)
 - [[```444fb40a```](https://github.com/jina-ai/docarray/commit/444fb40aef16307694c5adea496f64ad2dc15315)] __-__ add validate search fields (#1331) (*Saba Sturua*)
 - [[```02d66ed5```](https://github.com/jina-ai/docarray/commit/02d66ed593ddf5b9ea9ee69706b761ab9173908c)] __-__ save image tensor to file (#1335) (*Charlotte Gerhaher*)
 - [[```9b03d290```](https://github.com/jina-ai/docarray/commit/9b03d290284231eb861a6fbbd91d424a2f7e93c5)] __-__ implement document equality (#1329) (*Saba Sturua*)
 - [[```4e754567```](https://github.com/jina-ai/docarray/commit/4e754567b69da598cef67e7e0093befd646abb77)] __-__ add pil load (#1322) (*samsja*)
 - [[```90633c88```](https://github.com/jina-ai/docarray/commit/90633c88294103df3fefea7a29c67a54c7a8000c)] __-__ docarray fastapi simple integration (#1320) (*Saba Sturua*)
 - [[```d4277544```](https://github.com/jina-ai/docarray/commit/d4277544d8193d5735528d944d92bdb88ab5a4a2)] __-__ torch backend basic operation tests (#1306) (*Aman Agarwal*)
 - [[```a002bca2```](https://github.com/jina-ai/docarray/commit/a002bca2697ad931e78121da8cdc5578005b244a)] __-__ elasticsearch document index (#1196) (*Anne Yang*)
 - [[```6cd05f8b```](https://github.com/jina-ai/docarray/commit/6cd05f8b30ec90cd1e458955572a13315dfae00d)] __-__ implement push/pull interface from JAC, file and s3 (#1182) (*Jackmin801*)
 - [[```081a03f8```](https://github.com/jina-ai/docarray/commit/081a03f88685d97d3dd95f6ab21259e4ed4f7081)] __-__ __test__: DocumentArray method tests similar to list methods like reverse, sort, remove, pop (#1291) (*Aman Agarwal*)
 - [[```70fa4fa7```](https://github.com/jina-ai/docarray/commit/70fa4fa713325b8c3521b13fba0559abc8218bc7)] __-__ create documents from dict (#1283) (*Saba Sturua*)
 - [[```11d013e8```](https://github.com/jina-ai/docarray/commit/11d013e81833436b35a75331e5f3100abebac348)] __-__ add `get_paths()` instead of v1 `from_files()` (#1267) (*Charlotte Gerhaher*)
 - [[```38ee3a55```](https://github.com/jina-ai/docarray/commit/38ee3a55b141614b95dbee4ecb44a92c12d36a85)] __-__ add minimal logger (#1254) (*Saba Sturua*)
 - [[```e2c9b646```](https://github.com/jina-ai/docarray/commit/e2c9b64690d736f6d042cbec68c855fb2884beca)] __-__ __index__: index data with union types (#1220) (*Johannes Messner*)
 - [[```d8d36d77```](https://github.com/jina-ai/docarray/commit/d8d36d77429ac9f20192d7fb0c4645c3a20a5339)] __-__ ad user defined mapping for python type to db type (#1252) (*Charlotte Gerhaher*)
 - [[```ed4058cd```](https://github.com/jina-ai/docarray/commit/ed4058cdc321d26d026ccbd3823277066dfafe1c)] __-__ shift to mkdocs (#1244) (*samsja*)

### 🐞 Bug fixes

 - [[```f1d3ffd3```](https://github.com/jina-ai/docarray/commit/f1d3ffd399ddc2c02ba0edeae863b1b18f1bb91b)] __-__ ingore push pull test (#1403) (*samsja*)
 - [[```c45e3497```](https://github.com/jina-ai/docarray/commit/c45e34975007b6a8210683c189bb9b4d7341df1b)] __-__ fix push pull (#1384) (*samsja*)
 - [[```b13b869a```](https://github.com/jina-ai/docarray/commit/b13b869affe80aa470983dade6c7fee480576bb0)] __-__ needs succes all test before pushing to pypi prelease (#1388) (*samsja*)
 - [[```b2586e83```](https://github.com/jina-ai/docarray/commit/b2586e838d328c487389e1edfd8403c6b7438dfa)] __-__ add back cd from v1 (#1387) (*samsja*)
 - [[```8090d659```](https://github.com/jina-ai/docarray/commit/8090d659aa367a22aa3e44b16ec81b16d12d3a93)] __-__ add import checks and add weaviate doc index to index init (#1383) (*Charlotte Gerhaher*)
 - [[```f577fcb2```](https://github.com/jina-ai/docarray/commit/f577fcb2dd28afda34e29d2f04d3140b35757cd1)] __-__ fix install (#1382) (*samsja*)
 - [[```7956a7c2```](https://github.com/jina-ai/docarray/commit/7956a7c2070640f11c69fea6bf6f2acfd51fc372)] __-__ default dims=-1 for elastic index (#1368) (*Anne Yang*)
 - [[```7b472490```](https://github.com/jina-ai/docarray/commit/7b472490250728584d8d0b64424ce8cab5255272)] __-__ adjust remote file path in test and add toydata (#1364) (*Charlotte Gerhaher*)
 - [[```760c106f```](https://github.com/jina-ai/docarray/commit/760c106f184ca635d4938e0649bc591aa99fbf8e)] __-__ DocArrayStack to DocVec in docstrings (#1347) (*Aman Agarwal*)
 - [[```4d9ff9d6```](https://github.com/jina-ai/docarray/commit/4d9ff9d6504af9d0ea9dfd4624615879d1e6e37b)] __-__ in `.to_bytes()` return type specific bytes (#1341) (*Charlotte Gerhaher*)
 - [[```996f65f0```](https://github.com/jina-ai/docarray/commit/996f65f0b0a01d7ffda62b676e5c951c86931411)] __-__ fix type in es test (#1338) (*samsja*)
 - [[```9b7c3e22```](https://github.com/jina-ai/docarray/commit/9b7c3e2253b7eacd21b5a77182803a9e80b696c6)] __-__ return audio, image specific types in `.load()` and `.load_bytes()` (#1328) (*Charlotte Gerhaher*)
 - [[```9e03f082```](https://github.com/jina-ai/docarray/commit/9e03f0829d7432f07c7998aea014ba2148c76e08)] __-__ mark es test as index (#1314) (*samsja*)
 - [[```d73dc793```](https://github.com/jina-ai/docarray/commit/d73dc793d04b31802624d60c0d3f8335e119a55c)] __-__ flatten schema of abstract index (#1294) (*Anne Yang*)
 - [[```842664f8```](https://github.com/jina-ai/docarray/commit/842664f8059f2100cc0ab4f41aa9b3d25ca959cf)] __-__ remove files (#1305) (*samsja*)
 - [[```2602472d```](https://github.com/jina-ai/docarray/commit/2602472d338c78f8282291a49626b1827148c3d6)] __-__ docstring polish typing (#1299) (*samsja*)
 - [[```d86858d4```](https://github.com/jina-ai/docarray/commit/d86858d4855af53c724f1f3f031630a34a8d0f15)] __-__ rename DocArrayProto to DocumentArrayProto (#1297) (*samsja*)
 - [[```89c2a0a9```](https://github.com/jina-ai/docarray/commit/89c2a0a9c0fb3b1f5b8d300c3ce69ce0b6e38c92)] __-__ hnswlib doc index (#1277) (*Johannes Messner*)
 - [[```d0656f97```](https://github.com/jina-ai/docarray/commit/d0656f97afa181a55cc2d09393b1faede4cd39c8)] __-__ add int, float and others to doc summary (#1287) (*Charlotte Gerhaher*)
 - [[```33641271```](https://github.com/jina-ai/docarray/commit/33641271d94c72e6f84f76162f1a489f3a81c653)] __-__ proto ser and deser for nested tuple/dict/list (#1278) (*samsja*)
 - [[```df6dc0bd```](https://github.com/jina-ai/docarray/commit/df6dc0bd0a8138ba6682f65f04c327979e60eb43)] __-__ doc summary for dict and set attributes (#1279) (*Charlotte Gerhaher*)
 - [[```2af14cdf```](https://github.com/jina-ai/docarray/commit/2af14cdf74479e6f94240634b7a6fc873085a3c8)] __-__ bytes type in `TextDoc` and `VideoDoc` (#1270) (*Charlotte Gerhaher*)
 - [[```17892890```](https://github.com/jina-ai/docarray/commit/17892890ccd9a9e1eb5d6c7975d721bf2ec6ba5b)] __-__ disable pycharm da property detection (#1262) (*Charlotte Gerhaher*)
 - [[```32b3191d```](https://github.com/jina-ai/docarray/commit/32b3191dcb33506bde49202bf124a7dc470f58e1)] __-__ move test to integration test (#1260) (*samsja*)
 - [[```baebc194```](https://github.com/jina-ai/docarray/commit/baebc19437860131c8fb20a02b9b920e8eef4fb5)] __-__ up arrow emoji display in readme on docarray.org (#1034) (*Alex Cureton-Griffiths*)
 - [[```d594570f```](https://github.com/jina-ai/docarray/commit/d594570f90ebf47d1a2f1e0d39639f7a67334364)] __-__ __install docs__: consistency, grammar (#1028) (*Alex Cureton-Griffiths*)

### 🧼 Code Refactoring

 - [[```fc1af117```](https://github.com/jina-ai/docarray/commit/fc1af117861fbe141ceaa5f8733fe716d074eeb5)] __-__ change return type of find batched (#1339) (*Nikolas Pitsillos*)
 - [[```c48d6c04```](https://github.com/jina-ai/docarray/commit/c48d6c04a9a687ff4cea76b6f23bee67a067f525)] __-__ rename stack (#1376) (*samsja*)
 - [[```18ad2eab```](https://github.com/jina-ai/docarray/commit/18ad2eab1f0f7f0c972dcf434ef7ec3766fba2ee)] __-__ rename from_pandas to from_dataframe (#1358) (*samsja*)
 - [[```dffb227f```](https://github.com/jina-ai/docarray/commit/dffb227f71d0cc9e70ae635e3b40fc5eafbe869a)] __-__ remove warning, align naming (#1337) (*Johannes Messner*)
 - [[```a74b3bb4```](https://github.com/jina-ai/docarray/commit/a74b3bb43c4aa4a51cd6cf1ff534af80512c99dd)] __-__ rename DocArray to DocList (#1334) (*samsja*)
 - [[```221b440c```](https://github.com/jina-ai/docarray/commit/221b440c28bef432dc0caa408a2ddcdef6f719e0)] __-__ remove url validation (#1333) (*Saba Sturua*)
 - [[```69c7a77e```](https://github.com/jina-ai/docarray/commit/69c7a77ec6c9ed0d46dd548530b7690449527a86)] __-__ map_docs_batch to map_docs_batched (#1312) (*Charlotte Gerhaher*)
 - [[```d231f38a```](https://github.com/jina-ai/docarray/commit/d231f38a445fd243cb61fc0fe150612169319ddd)] __-__ rename `Document` to `Doc` (#1293) (*samsja*)
 - [[```b029b5e6```](https://github.com/jina-ai/docarray/commit/b029b5e62c8b7a9b2f9fd0e92f8e0a5789b4428f)] __-__ bytes to bytes_ in predefined documents (#1273) (*Charlotte Gerhaher*)
 - [[```64532dd8```](https://github.com/jina-ai/docarray/commit/64532dd883b4195d3fb2c713620532a52e3707af)] __-__ __da__: remove tensor type from `DocumentArray` init (#1268) (*samsja*)
 - [[```6707f4c9```](https://github.com/jina-ai/docarray/commit/6707f4c975b3ccbdc20af96a79b694ec15f5a66a)] __-__ doc index structure (#1266) (*Saba Sturua*)
 - [[```d9d7bd78```](https://github.com/jina-ai/docarray/commit/d9d7bd78760d008a762cd8bee29c9b8e30725fdd)] __-__ rename filter to filter_docs to avoid shadowing of filtern (#1257) (*Charlotte Gerhaher*)

### 📗 Documentation

 - [[```21da4625```](https://github.com/jina-ai/docarray/commit/21da4625332c68a81c829f261a189cb5a684b5ea)] __-__ add migration guide (#1398) (*Charlotte Gerhaher*)
 - [[```5729a62a```](https://github.com/jina-ai/docarray/commit/5729a62a18b5b88292fb865848c9523cccbe5e76)] __-__ polish docs (#1400) (*samsja*)
 - [[```0872b6b9```](https://github.com/jina-ai/docarray/commit/0872b6b984ea40c413d639ab6152f5812241cc66)] __-__ final round of fixes (#1401) (*Alex Cureton-Griffiths*)
 - [[```2a02635e```](https://github.com/jina-ai/docarray/commit/2a02635ee9965193720e281c0010115b923bef7a)] __-__ document index (#1346) (*Nan Wang*)
 - [[```091004b1```](https://github.com/jina-ai/docarray/commit/091004b18479a179ed9d16fec4b2a54dad65b0d4)] __-__ fix style in collapsible box (#1396) (*Anne Yang*)
 - [[```baa3cbc3```](https://github.com/jina-ai/docarray/commit/baa3cbc39b66dffe64e8f07ea070bec6c152daea)] __-__ resolve todos and incomplete sections and missing links (#1394) (*Charlotte Gerhaher*)
 - [[```ce3fb6ee```](https://github.com/jina-ai/docarray/commit/ce3fb6ee9c2b66951884556e3c83517362adbe46)] __-__ fix doc store code snippet in README.md (#1389) (*Charlotte Gerhaher*)
 - [[```85d41ef6```](https://github.com/jina-ai/docarray/commit/85d41ef60a39f9d5bb9de06c286ec154020702a2)] __-__ fix represent, intro (#1390) (*Alex Cureton-Griffiths*)
 - [[```df126173```](https://github.com/jina-ai/docarray/commit/df12617389375d47b898f92204c585e345855091)] __-__ adjust file paths to main branch (#1385) (*Charlotte Gerhaher*)
 - [[```dba02b3e```](https://github.com/jina-ai/docarray/commit/dba02b3ecc702e1e6e4827099a77b66e3a405370)] __-__ check docstrings and clean up (#1373) (*Charlotte Gerhaher*)
 - [[```42523c7b```](https://github.com/jina-ai/docarray/commit/42523c7bdcc035cb57f40607ddd0b5a0b8995031)] __-__ tweak the readme (#1372) (*Johannes Messner*)
 - [[```13c97fe1```](https://github.com/jina-ai/docarray/commit/13c97fe1bdeb6b344e6ec19031f0034a0dd0f4a9)] __-__ add storing with file (#1348) (*Nan Wang*)
 - [[```6c45d4b1```](https://github.com/jina-ai/docarray/commit/6c45d4b1eb0722462751eaef70e4f0cefdf079c1)] __-__ add missing links and clean up (#1370) (*Charlotte Gerhaher*)
 - [[```18f2a3f9```](https://github.com/jina-ai/docarray/commit/18f2a3f963715dcdc98c58b78ffd673df4914ec3)] __-__ add sending section (#1350) (*Nan Wang*)
 - [[```f5b0ea32```](https://github.com/jina-ai/docarray/commit/f5b0ea3231722e48ccea357782a6d0e58c11e85d)] __-__ __menu__: consistency, wording fixes (#1363) (*Alex Cureton-Griffiths*)
 - [[```9a13f93c```](https://github.com/jina-ai/docarray/commit/9a13f93c6c3841c926a9bb5a0f1230f1a444cb1c)] __-__ fastapi integration section (#1326) (*Saba Sturua*)
 - [[```b8f178ee```](https://github.com/jina-ai/docarray/commit/b8f178ee8a7daa418709197b0161a7dc560a500e)] __-__ multi modalities (#1317) (*Charlotte Gerhaher*)
 - [[```2f711438```](https://github.com/jina-ai/docarray/commit/2f711438db6002fc08cdea3f7b220a09b9af1400)] __-__ add DocList and DocVec section (#1343) (*samsja*)
 - [[```f1bb4e4b```](https://github.com/jina-ai/docarray/commit/f1bb4e4bb7482f9d7a221ee6ec8ae4d36cd3b727)] __-__ rewrite readme (#1340) (*Johannes Messner*)
 - [[```0e18796f```](https://github.com/jina-ai/docarray/commit/0e18796f7996d12da10bf94d6632b5004bfcb8a9)] __-__ add torch and tf tensors to `Api references` section (#1345) (*Charlotte Gerhaher*)
 - [[```a6f04fbb```](https://github.com/jina-ai/docarray/commit/a6f04fbb128e6de69489bab2cadbeaf0f735cb5a)] __-__ add audio2text showcase (#1336) (*Aziz Belaweid*)
 - [[```c0718e57```](https://github.com/jina-ai/docarray/commit/c0718e572c47088d5c55a25f6d27f2f2d9697038)] __-__ add user guide (#1292) (*samsja*)
 - [[```2aa7727b```](https://github.com/jina-ai/docarray/commit/2aa7727b7681c978ce0fe7246e8c8fd0e9d307c0)] __-__ fix map docstring (#1311) (*samsja*)
 - [[```f3948bd3```](https://github.com/jina-ai/docarray/commit/f3948bd37834905708297875481060cc5a138a23)] __-__ fix docstring example of find_batched (#1308) (*Johannes Messner*)
 - [[```4cba12fc```](https://github.com/jina-ai/docarray/commit/4cba12fcc8771a580870d1ab7664d8dbbaeab773)] __-__ add utils section (#1307) (*samsja*)
 - [[```9728bd5b```](https://github.com/jina-ai/docarray/commit/9728bd5bfb6e82de0a89e03ee2907fdd4057a4b2)] __-__ fix up english (#1285) (*Alex Cureton-Griffiths*)
 - [[```a92c89b1```](https://github.com/jina-ai/docarray/commit/a92c89b1da395e266e2d7d760ddbb374f5f7ef70)] __-__ add explanation about id field (#1242) (*Johannes Messner*)
 - [[```9c95f3ad```](https://github.com/jina-ai/docarray/commit/9c95f3adfd329f1b11671ca3b9e8c10163021206)] __-__ remove docsqa (#1214) (*AlaeddineAbdessalem*)
 - [[```13c8f149```](https://github.com/jina-ai/docarray/commit/13c8f149eb296b4531bc1489e69a40f979884ccd)] __-__ fix opensearch docker compose yaml file (#997) (*AlaeddineAbdessalem*)
 - [[```1fec2c24```](https://github.com/jina-ai/docarray/commit/1fec2c24dfc406448c76185e700d805eaa67adb6)] __-__ fixes for graphql, torch pages (#1205) (*Alex Cureton-Griffiths*)
 - [[```5de8fd45```](https://github.com/jina-ai/docarray/commit/5de8fd4519020e3cf6ec16911982059548f69b08)] __-__ fix small typo in pull (#1158) (*Jackmin801*)
 - [[```fb72abd2```](https://github.com/jina-ai/docarray/commit/fb72abd267f0fcfd744f3ebb7ea738b47f68ae46)] __-__ remove mention about future support for big data loading (#1118) (*Tanguy Abel*)
 - [[```34403f7c```](https://github.com/jina-ai/docarray/commit/34403f7c1de7cf73ccc9871e39d6cddd6b69e158)] __-__ fix urls to docs.docarray.org (#1101) (*Alex Cureton-Griffiths*)
 - [[```dda41063```](https://github.com/jina-ai/docarray/commit/dda410634dc096ee1132185012647f8f84dff5df)] __-__ scipy install (#1086) (*Johannes Messner*)
 - [[```0a2b2736```](https://github.com/jina-ai/docarray/commit/0a2b2736660052ab04fe73af09ac38e98244cddb)] __-__ remove ecosystem from docs (#1092) (*Joan Fontanals*)
 - [[```5beeaaa7```](https://github.com/jina-ai/docarray/commit/5beeaaa778ad29175afdd95bdb1ff781b64d9c1e)] __-__ add calendar link to readme (#1060) (*Johannes Messner*)

### 🍹 Other Improvements

 - [[```56e11a3c```](https://github.com/jina-ai/docarray/commit/56e11a3c431bcd051281a01d91d1086a0eacdf90)] __-__ install poetry during release (#1404) (*samsja*)
 - [[```aa346385```](https://github.com/jina-ai/docarray/commit/aa346385852e23c9d52a747123f6d5d73fc606b5)] __-__ add release template (#1402) (*samsja*)
 - [[```a32e8e59```](https://github.com/jina-ai/docarray/commit/a32e8e595be7fa7d0d10201adba959d98471b67e)] __-__ fix install (#1399) (*samsja*)
 - [[```2452c72f```](https://github.com/jina-ai/docarray/commit/2452c72fc99df0278a0e2c0f8e6f1a3326c8b3df)] __-__ fix update of pyproject toml version (#1395) (*samsja*)
 - [[```3ab98f89```](https://github.com/jina-ai/docarray/commit/3ab98f894ff4b1f882c5827dc298b931807a48e3)] __-__ fix ci to do pre-release (#1393) (*samsja*)
 - [[```30dee5a8```](https://github.com/jina-ai/docarray/commit/30dee5a823ec025bff0f3128b3d7a234eed9901d)] __-__ fix cd not running (#1392) (*samsja*)
 - [[```570577dc```](https://github.com/jina-ai/docarray/commit/570577dcfbdb70a73177aee3092652eb56d78375)] __-__ fix readme (#1386) (*samsja*)
 - [[```aef8b626```](https://github.com/jina-ai/docarray/commit/aef8b626876eed1aaf27c3770efc4b694595c98c)] __-__ make weaviate dependency optional (#1381) (*Johannes Messner*)
 - [[```6e9510e6```](https://github.com/jina-ai/docarray/commit/6e9510e6e650a87d957d38e811230076aa566647)] __-__ fix logo readme (#1380) (*samsja*)
 - [[```dbf2bba3```](https://github.com/jina-ai/docarray/commit/dbf2bba3c74169e58ef82fa1e3e46f492cab95f7)] __-__ prepare to merge to main branch (#1377) (*samsja*)
 - [[```0f94d1a5```](https://github.com/jina-ai/docarray/commit/0f94d1a51c907bb91031eac3332969bc81da39ae)] __-__ ci use different worker for each DocIndex (#1375) (*samsja*)
 - [[```fb6b02bc```](https://github.com/jina-ai/docarray/commit/fb6b02bc434d99a785d8ec9d3c05160db6a079a9)] __-__ fix mypy (#1371) (*samsja*)
 - [[```3f739aef```](https://github.com/jina-ai/docarray/commit/3f739aef1319e567a236fe4b5c0d65e0251ece80)] __-__ add instructions to pip installs and group extras (#1281) (*Charlotte Gerhaher*)
 - [[```d5e87cd7```](https://github.com/jina-ai/docarray/commit/d5e87cd7c16ed7bbe854655a142540c9083d3874)] __-__ add docstring test (#1298) (*samsja*)
 - [[```5d48be80```](https://github.com/jina-ai/docarray/commit/5d48be801b9b418bdb82ff9bf9af59656fe9c91e)] __-__ __docs__: add ci and fix docs ui (#1295) (*samsja*)
 - [[```bd34acd8```](https://github.com/jina-ai/docarray/commit/bd34acd8e92109401adcf1897ae0f4cf61137e63)] __-__ __version__: the next version will be 0.21.1 (*Jina Dev Bot*)
 - [[```ca2973f0```](https://github.com/jina-ai/docarray/commit/ca2973f03f8209e5d1f831bf08e84dff033322d7)] __-__ bump version (#1023) (*Johannes Messner*)

