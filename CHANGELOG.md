














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

<a name=release-note-0-31-0></a>
## Release Note (`0.31.0`)

> Release time: 2023-05-08 09:41:08



🙇 We'd like to thank all contributors for this new release! In particular,
 samsja,  Joan Fontanals,  Charlotte Gerhaher,  Carren,  Aman Agarwal,  カレン,  Johannes Messner,  Alex Cureton-Griffiths,  Anne Yang,  Yanlong Wang,  Scott Martens,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```eb693183```](https://github.com/jina-ai/docarray/commit/eb693183aa68ebf3bf0950d1eca1c6229599a5cc)] __-__ index or collection name will default to doc-type name (#1486) (*Joan Fontanals*)
 - [[```c4c4b9f8```](https://github.com/jina-ai/docarray/commit/c4c4b9f86456072b1488f0bc0f53610bf2a46d48)] __-__ add to_json alias (#1494) (*Joan Fontanals*)
 - [[```b3649b4b```](https://github.com/jina-ai/docarray/commit/b3649b4b0f020887397f96e757711f71095b7fee)] __-__ make DocList an actual Python List (#1457) (*Joan Fontanals*)
 - [[```a25d1ae2```](https://github.com/jina-ai/docarray/commit/a25d1ae23df6d6201530be93e93ddfa8f2c27412)] __-__ store as json-array (#1459) (*Carren*)
 - [[```f65e0231```](https://github.com/jina-ai/docarray/commit/f65e02317bf3d5d3e7862369bbd829f18bfd94e3)] __-__ add in-memory doc index (#1441) (*Charlotte Gerhaher*)
 - [[```83e73847```](https://github.com/jina-ai/docarray/commit/83e7384746465d8bf9a2275444d98bae04948114)] __-__ add config to load more field that the schema (#1437) (*samsja*)
 - [[```9bf0512a```](https://github.com/jina-ai/docarray/commit/9bf0512aa28ac2535ef21f1a9f1e27cd4e8b7e21)] __-__ add len into DocIndex (#1454) (*カレン*)
 - [[```2382ced9```](https://github.com/jina-ai/docarray/commit/2382ced914bc0ea197906adbfa7c06d3ad4dc6ff)] __-__ add google analytics (#1432) (*samsja*)
 - [[```4fac9a88```](https://github.com/jina-ai/docarray/commit/4fac9a88504e88ed69fb05a05d804cfaf6da220b)] __-__ add install instructions for hnswlib and elastic doc index (#1431) (*Charlotte Gerhaher*)
 - [[```53304cef```](https://github.com/jina-ai/docarray/commit/53304cef16b65ce2f4819baa4a72b7c0321ebfb2)] __-__ point to older versions when importing Document or DocumentArray (#1422) (*Charlotte Gerhaher*)

### 🐞 Bug fixes

 - [[```67c7e6d5```](https://github.com/jina-ai/docarray/commit/67c7e6d578a0e7a1566a14362b2e5bf3aeb46f13)] __-__ fix nested doc to json (#1502) (*samsja*)
 - [[```f5f692d8```](https://github.com/jina-ai/docarray/commit/f5f692d8d23c9c91ce6afe040cb34b0490e10aa8)] __-__ fix extend with itself infinite recursion (#1493) (*Joan Fontanals*)
 - [[```b77df16b```](https://github.com/jina-ai/docarray/commit/b77df16be93b33438941290be7dba184ec119912)] __-__ allow doclist to have nested optional document (#1472) (*samsja*)
 - [[```e37a59e8```](https://github.com/jina-ai/docarray/commit/e37a59e865fec86ca8b7f659ac03c88172d12f30)] __-__ fix to dict exclude (#1481) (*samsja*)
 - [[```baaf5cba```](https://github.com/jina-ai/docarray/commit/baaf5cbad4dcf2b7ae62ad1d44f8e7fdf8c7549b)] __-__ use `pd.concat()` instead `df.append()` in `to_dataframe()` to avoid warning (#1478) (*Charlotte Gerhaher*)
 - [[```deb81b93```](https://github.com/jina-ai/docarray/commit/deb81b9301af763b395e3fc839f81a39dd58b075)] __-__ fix equal of BaseDoc with nested DocList (#1477) (*samsja*)
 - [[```d497cba4```](https://github.com/jina-ai/docarray/commit/d497cba4f6654b399331785b5424ded136440fec)] __-__ return doclist of same type as input index in find and findbatched (#1470) (*Charlotte Gerhaher*)
 - [[```815ba889```](https://github.com/jina-ai/docarray/commit/815ba889328d133ab198ef8ff23431d49da02c21)] __-__ from proto for nested List and Dict (#1471) (*samsja*)
 - [[```9cfe2a6d```](https://github.com/jina-ai/docarray/commit/9cfe2a6d4f860efe74a5651d78d5eefeb87a07de)] __-__ raise error when calling to csv or to dataframe on non homegenou… (#1467) (*samsja*)
 - [[```0ed8523e```](https://github.com/jina-ai/docarray/commit/0ed8523e1fa91d0f90b2f597ce588880f3e3f095)] __-__ len check in doc index logger if docs isinstance of BaseDoc (#1465) (*Charlotte Gerhaher*)
 - [[```7febaca0```](https://github.com/jina-ai/docarray/commit/7febaca086a7622c00d3d85e93d459fd59d8310f)] __-__ add logs to elasticsearch index  (#1427) (*Aman Agarwal*)
 - [[```77c9d7bc```](https://github.com/jina-ai/docarray/commit/77c9d7bc4165e82ab76027ecec41fa60197dbf3f)] __-__ docindex urls (#1433) (*Alex Cureton-Griffiths*)
 - [[```52960981```](https://github.com/jina-ai/docarray/commit/5296098168ca43bd8925008ea88be32b62a61068)] __-__ torch tensor with grad to numpy (#1429) (*Johannes Messner*)
 - [[```0c73ad83```](https://github.com/jina-ai/docarray/commit/0c73ad83cb7c65d4d11d23d021f3661655765dd7)] __-__ passes max_element when load index in hnswlib (#1426) (*Anne Yang*)
 - [[```bf2e50c2```](https://github.com/jina-ai/docarray/commit/bf2e50c2ac6e63ad6e7f098c747a6fe8eab52150)] __-__ binary format version bump (#1414) (*Yanlong Wang*)
 - [[```f67508ad```](https://github.com/jina-ai/docarray/commit/f67508ad7d00001eea2af7763c806c6a08263736)] __-__ save index during creation for hnswlib (#1424) (*Anne Yang*)
 - [[```98dfe69e```](https://github.com/jina-ai/docarray/commit/98dfe69e71a78d60628dfecaa991bd4956a49d62)] __-__ install commands after removing of common (#1421) (*Charlotte Gerhaher*)
 - [[```427d2a77```](https://github.com/jina-ai/docarray/commit/427d2a775898c6f202e379345e3a28aeaf808dfe)] __-__ skip push pull (#1405) (*samsja*)

### 🧼 Code Refactoring

 - [[```7ba430ce```](https://github.com/jina-ai/docarray/commit/7ba430cec5d022af0ba621bbe9fe81d26492de8b)] __-__ rename InMemoryDocIndex to InMemoryExactNNIndex (#1466) (*Charlotte Gerhaher*)

### 📗 Documentation

 - [[```091b1802```](https://github.com/jina-ai/docarray/commit/091b1802ca3ff7b7bcc71d5425f7fae1ad1f43ba)] __-__ change cosine similarity to distance for hnswlib doc index (#1476) (*Charlotte Gerhaher*)
 - [[```a76cc615```](https://github.com/jina-ai/docarray/commit/a76cc6159ab1aaf6932b46e639a6c2396ef751f5)] __-__ fix fastapi (#1453) (*samsja*)
 - [[```a36522ce```](https://github.com/jina-ai/docarray/commit/a36522cecd2215d2b4b96a63e8381018aff85121)] __-__ fix typos (#1436) (*Johannes Messner*)
 - [[```fad1290e```](https://github.com/jina-ai/docarray/commit/fad1290e0137f7105cbb0e9cd3b7bfb28c8c086f)] __-__ index predefined documents (#1434) (*Johannes Messner*)
 - [[```ade63830```](https://github.com/jina-ai/docarray/commit/ade6383025e41b6a769e1d915b051fe85bb18dc3)] __-__ __storage__: proofread (#1423) (*Alex Cureton-Griffiths*)
 - [[```3eb7511f```](https://github.com/jina-ai/docarray/commit/3eb7511fdc00d0eef1c6d753af28085d60f5eab1)] __-__ __contributing__: basic fixes (#1418) (*Alex Cureton-Griffiths*)
 - [[```70924065```](https://github.com/jina-ai/docarray/commit/70924065d20a5396010f3518b3a9cb5770038e4f)] __-__ clean up data types section (#1412) (*Charlotte Gerhaher*)
 - [[```ef527c58```](https://github.com/jina-ai/docarray/commit/ef527c58cb8350ba0e34cd347a7750e7c7c4041a)] __-__ consistent wording (#1419) (*Alex Cureton-Griffiths*)
 - [[```cdcdcf3e```](https://github.com/jina-ai/docarray/commit/cdcdcf3e4c150f09dbc2e7b644cd014d79b14983)] __-__ __migration-guide__: fix issues (#1417) (*Alex Cureton-Griffiths*)
 - [[```4e04c069```](https://github.com/jina-ai/docarray/commit/4e04c06974291cf27af0af43c864d2fd16e369e1)] __-__ fix README.md (#1411) (*Scott Martens*)
 - [[```176fd36f```](https://github.com/jina-ai/docarray/commit/176fd36fac451f5cb89490c13890982a525e1453)] __-__ remove duplicate api reference section (#1408) (*Johannes Messner*)

### 🏁 Unit Test and CICD

 - [[```2d426df6```](https://github.com/jina-ai/docarray/commit/2d426df6ce182b4533c9db35a5e2fbc1ef4b8822)] __-__ try to get jac tests back up (#1485) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```d1f13d68```](https://github.com/jina-ai/docarray/commit/d1f13d6818102f2ba79b9130b22df96435c7446b)] __-__ remove file (#1499) (*samsja*)
 - [[```c9d0f719```](https://github.com/jina-ai/docarray/commit/c9d0f719864b73340a6610bc25e642268fc24b36)] __-__ ci fix jac by passing secret to workflow (#1492) (*samsja*)
 - [[```ca6c0e22```](https://github.com/jina-ai/docarray/commit/ca6c0e2287c7dc1eaf5e828ba0c2b9b228526b48)] __-__ add seperate jac ci (#1487) (*samsja*)
 - [[```8df9e7f3```](https://github.com/jina-ai/docarray/commit/8df9e7f3ae700ee84f766d38fd4cc2fb63908567)] __-__ bump version to 0.31.0 (#1458) (*samsja*)
 - [[```46e55f98```](https://github.com/jina-ai/docarray/commit/46e55f98b293cdb0fc06fdd60898d29efd7e3545)] __-__ use temp file for docs build action (#1416) (*samsja*)
 - [[```fb796680```](https://github.com/jina-ai/docarray/commit/fb796680f9c2128ee52e42b54141076412fc4103)] __-__ fix cd docs, readme and release  (#1413) (*samsja*)
 - [[```ce89d5b7```](https://github.com/jina-ai/docarray/commit/ce89d5b7a957552bcf97a74758a5cb4f4f4c7dfd)] __-__ bump pyproject version (#1415) (*samsja*)
 - [[```e6b3700d```](https://github.com/jina-ai/docarray/commit/e6b3700ddaf9e9b648d8c92b4dd2024af064c790)] __-__ __version__: the next version will be 0.30.1 (*Jina Dev Bot*)
 - [[```56e11a3c```](https://github.com/jina-ai/docarray/commit/56e11a3c431bcd051281a01d91d1086a0eacdf90)] __-__ install poetry during release (#1404) (*samsja*)

<a name=release-note-0-31-1></a>
## Release Note (`0.31.1`)

> Release time: 2023-05-08 16:30:54



🙇 We'd like to thank all contributors for this new release! In particular,
 samsja,  Jina Dev Bot,  🙇


### 🐞 Bug fixes

 - [[```302b49a9```](https://github.com/jina-ai/docarray/commit/302b49a99f95f9449e131f5fcc176dad12d143df)] __-__ json and dict when calling empty doclist (#1512) (*samsja*)

### 🧼 Code Refactoring

 - [[```ebecff85```](https://github.com/jina-ai/docarray/commit/ebecff85c97771184a5ec1acac56f274e452fdde)] __-__ remove useless step in json encoder (#1513) (*samsja*)

### 🍹 Other Improvements

 - [[```8c7eb95f```](https://github.com/jina-ai/docarray/commit/8c7eb95fac2b5064fc90bd59c25fa2de0df9508e)] __-__ bump version (#1514) (*samsja*)
 - [[```94cccbb5```](https://github.com/jina-ai/docarray/commit/94cccbb5fb296e7b2639ecec12c0f163f06546d6)] __-__ fix cd for documentation (#1509) (*samsja*)
 - [[```04b930f4```](https://github.com/jina-ai/docarray/commit/04b930f4751a44188b6cb531c1cb4d2f5c43ae02)] __-__ __version__: the next version will be 0.31.1 (*Jina Dev Bot*)

<a name=release-note-0-32-0></a>
## Release Note (`0.32.0`)

> Release time: 2023-05-16 11:29:36



🙇 We'd like to thank all contributors for this new release! In particular,
 samsja,  Saba Sturua,  Anne Yang,  Zhaofeng Miao,  Mohammad Kalim Akram,  Kacper Łukawski,  Joan Fontanals,  Johannes Messner,  IyadhKhalfallah,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```db4cc1a7```](https://github.com/jina-ai/docarray/commit/db4cc1a770f29368683e6b2be094e4062972f3ca)] __-__ save and load inmemory index (#1534) (*Saba Sturua*)
 - [[```44977b75```](https://github.com/jina-ai/docarray/commit/44977b75ee9c96a272637ebf21a14285b1c6aeab)] __-__ subindex for document index (#1428) (*Anne Yang*)
 - [[```096f6449```](https://github.com/jina-ai/docarray/commit/096f644980e63391e3ca7bfc8486b877b181b07f)] __-__ search_field  should be optional in hybrid text search (#1516) (*Anne Yang*)
 - [[```e9fc57af```](https://github.com/jina-ai/docarray/commit/e9fc57af6b7fb0a70534971a8f6a2062be444f94)] __-__ openapi tensor shapes (#1510) (*Johannes Messner*)

### 🐞 Bug fixes

 - [[```2d3bcd2e```](https://github.com/jina-ai/docarray/commit/2d3bcd2ef2886e938cd1d47a7cebb5bc5cb14b34)] __-__ check if filepath exists for inmemory index (#1537) (*Saba Sturua*)
 - [[```1f2dcea6```](https://github.com/jina-ai/docarray/commit/1f2dcea6e91140de9dc84b251604197dfbae4e53)] __-__ add empty judgement to index search (#1533) (*Anne Yang*)
 - [[```8dd050f5```](https://github.com/jina-ai/docarray/commit/8dd050f554f44f0f79aa5b8ce19948848940a283)] __-__ detach the torch tensors (#1526) (*Mohammad Kalim Akram*)
 - [[```bfebad6b```](https://github.com/jina-ai/docarray/commit/bfebad6be30acd24bd4b9ff92c30522fc803c87c)] __-__ DocVec display (#1522) (*Anne Yang*)
 - [[```cdde136c```](https://github.com/jina-ai/docarray/commit/cdde136c8002a9a114945648e6d4e135b15a14cd)] __-__ docs link (#1518) (*IyadhKhalfallah*)

### 📗 Documentation

 - [[```02afeb9f```](https://github.com/jina-ai/docarray/commit/02afeb9f6be25406beb2f6480fde8954729a272b)] __-__ __store_jac__: remove wrong info (#1531) (*Zhaofeng Miao*)
 - [[```bc7f7253```](https://github.com/jina-ai/docarray/commit/bc7f72533cbe2af18fd99ea2bfe393ec4157e2de)] __-__ fix link to documentation in readme (#1525) (*Joan Fontanals*)
 - [[```0ef6772d```](https://github.com/jina-ai/docarray/commit/0ef6772d35e2a4747e0fd7c39384424a79f8ec62)] __-__ flatten structure (#1520) (*Johannes Messner*)

### 🍹 Other Improvements

 - [[```9657d6fb```](https://github.com/jina-ai/docarray/commit/9657d6fbd69cc0f3686da9b056989fe8ce23cd00)] __-__ bump to 0.32.0 (#1541) (*samsja*)
 - [[```9705431b```](https://github.com/jina-ai/docarray/commit/9705431b8937a090924fe875f5550dc926802315)] __-__ Add more Qdrant examples (#1527) (*Kacper Łukawski*)
 - [[```445a72fa```](https://github.com/jina-ai/docarray/commit/445a72fa2a32c30de801f4d3203866d85de23990)] __-__ add protobuf in the hnsw extra (#1524) (*Saba Sturua*)
 - [[```20fdcd27```](https://github.com/jina-ai/docarray/commit/20fdcd27b786f7fea9ab4bc4ee5bbbd244ba66f0)] __-__ __version__: the next version will be 0.31.2 (*Jina Dev Bot*)

<a name=release-note-0-32-1></a>
## Release Note (`0.32.1`)

> Release time: 2023-05-26 14:50:34



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  maxwelljin,  Johannes Messner,  aman-exp-infy,  Saba Sturua,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```8651e6e8```](https://github.com/jina-ai/docarray/commit/8651e6e88cef3f66c6a6eeca0531e26e2b4ca18d)] __-__ logs added for es8 index (#1551) (*aman-exp-infy*)

### 🐞 Bug fixes

 - [[```5d41c13c```](https://github.com/jina-ai/docarray/commit/5d41c13c96de299ac8035fd09d3bdd32dc518036)] __-__ fix None embedding exact nn search (#1575) (*Joan Fontanals*)
 - [[```7a7a83a5```](https://github.com/jina-ai/docarray/commit/7a7a83a5d7526b8840e4b98c966cdbc635280bbc)] __-__ support list in document class (#1557) (#1569) (*maxwelljin*)
 - [[```40549f4a```](https://github.com/jina-ai/docarray/commit/40549f4aeacde1522fb6a3406c98b8dbd14e0858)] __-__ fix anydoc deserialization (#1571) (*Joan Fontanals*)
 - [[```44317570```](https://github.com/jina-ai/docarray/commit/44317570395380bcdff8d7de9e815b42460f5b9c)] __-__ dict method for document view (#1559) (*Johannes Messner*)

### 🧼 Code Refactoring

 - [[```0bcc956d```](https://github.com/jina-ai/docarray/commit/0bcc956da6d9d4971ee0b92f69fa776d7aae24f1)] __-__ uncaped tests as a nightly job (#1540) (*Saba Sturua*)

### 📗 Documentation

 - [[```0e6aa3b6```](https://github.com/jina-ai/docarray/commit/0e6aa3b6f43d1762500af96b646e538af44be1b5)] __-__ update doc building guide (#1566) (*Johannes Messner*)
 - [[```a01a0554```](https://github.com/jina-ai/docarray/commit/a01a05542d17264b8a164bec783633658deeedb8)] __-__ explain the state of doclist in fastapi (#1546) (*Johannes Messner*)

### 🍹 Other Improvements

 - [[```8a2e92a3```](https://github.com/jina-ai/docarray/commit/8a2e92a3f94efc77d90e0747c246bdcf2ce72dfd)] __-__ update pyproject.toml (#1581) (*Joan Fontanals*)
 - [[```9b5cbeda```](https://github.com/jina-ai/docarray/commit/9b5cbedaa43ea392b985e0ad293839523ce57030)] __-__ __version__: the next version will be 0.32.1 (*Jina Dev Bot*)

<a name=release-note-0-33-0></a>
## Release Note (`0.33.0`)

> Release time: 2023-06-06 14:05:56



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  Saba Sturua,  samsja,  maxwelljin,  Mohammad Kalim Akram,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```110f714f```](https://github.com/jina-ai/docarray/commit/110f714fda40689c4c743bc825bd1e017a739d9d)] __-__ avoid stack embedding for every search (#1586) (*maxwelljin*)
 - [[```5e74fcca```](https://github.com/jina-ai/docarray/commit/5e74fcca1ef0ef03f823b888b8585c3a0177144e)] __-__ tensor coersion (#1588) (*samsja*)

### 🐞 Bug fixes

 - [[```f7371b48```](https://github.com/jina-ai/docarray/commit/f7371b48df7e93de4808a9cbfcf0c89420a12129)] __-__ filter limits (#1618) (*Saba Sturua*)
 - [[```6903c773```](https://github.com/jina-ai/docarray/commit/6903c773490459a002fad2751dd3238b735e8d5e)] __-__ hnswlib must be able to search with limit more than num docs (#1611) (*Joan Fontanals*)
 - [[```e24be8d6```](https://github.com/jina-ai/docarray/commit/e24be8d6eff48baef440d184171a0c2e3356d0bf)] __-__ allow update on HNSWLibIndex (#1604) (*Joan Fontanals*)
 - [[```3cef708c```](https://github.com/jina-ai/docarray/commit/3cef708cce24aaa017fb2a5af5aed33bc029df09)] __-__ dynamically resize internal index to adapt to increasing number of docs (#1602) (*Joan Fontanals*)
 - [[```a5c90064```](https://github.com/jina-ai/docarray/commit/a5c90064cb2da0120ee6bdfcec613ef8f1447596)] __-__ fix simple usage of HNSWLib (#1596) (*Joan Fontanals*)
 - [[```88414ce2```](https://github.com/jina-ai/docarray/commit/88414ce25ec1808565f6f46d9051a08646f765b4)] __-__ fix InMemoryExactNN index initialization with nested DocList (#1582) (*Joan Fontanals*)
 - [[```5d0e24c9```](https://github.com/jina-ai/docarray/commit/5d0e24c94457e04f3e902b6ecd33d4e607c7636e)] __-__ fix summary of Doc with list (#1595) (*Joan Fontanals*)
 - [[```f765d9f4```](https://github.com/jina-ai/docarray/commit/f765d9f4f01089d0ab361cecd771efc97970ce92)] __-__ solve issues caused by issubclass (#1594) (*maxwelljin*)
 - [[```5ee87876```](https://github.com/jina-ai/docarray/commit/5ee878763cbc35f95d26a0fc3842211d8add3e16)] __-__ make example payload a string and not bytes (#1587) (*Joan Fontanals*)

### 🧼 Code Refactoring

 - [[```f9e504ef```](https://github.com/jina-ai/docarray/commit/f9e504efbc7229dd5d29d4fecef7f9d0bfb3dbc9)] __-__ minor changes in weaviate (#1621) (*Saba Sturua*)
 - [[```4eec5599```](https://github.com/jina-ai/docarray/commit/4eec5599bf2905a191e4cc5614a1813e91f4c02f)] __-__ make AnyTensor a class (#1552) (*Mohammad Kalim Akram*)

### 📗 Documentation

 - [[```29c2d23a```](https://github.com/jina-ai/docarray/commit/29c2d23a618704e9ed13108c754e09c6ef053a93)] __-__ add forward declaration steps to example to avoid pickling error (#1615) (*Joan Fontanals*)
 - [[```5e6bf755```](https://github.com/jina-ai/docarray/commit/5e6bf7550bf2395244633c4a19b7d812b7d6fe9d)] __-__ fix n_dim to dim in docs (#1610) (*Joan Fontanals*)
 - [[```de8c654b```](https://github.com/jina-ai/docarray/commit/de8c654bd2f4c5465943c974520021e39ff07ab4)] __-__ add in memory to documentation as list of supported vector index (#1607) (*Joan Fontanals*)
 - [[```1e41b5c5```](https://github.com/jina-ai/docarray/commit/1e41b5c59e4f2c7d14de1619141ed35898bbc815)] __-__ add a tensor section to docs (#1576) (*samsja*)

### 🍹 Other Improvements

 - [[```68194f49```](https://github.com/jina-ai/docarray/commit/68194f492a84ecfb61ffda1b669debe156a24a37)] __-__ update version to 0.33 (#1626) (*Joan Fontanals*)
 - [[```ac2e417e```](https://github.com/jina-ai/docarray/commit/ac2e417e9fc23ac06ebed515de0b0688827c145a)] __-__ fix issue template (#1624) (*samsja*)
 - [[```e1777144```](https://github.com/jina-ai/docarray/commit/e177714491caaa28dd1990db52ce3359416b8ab0)] __-__ add a better looking issue template (#1623) (*samsja*)
 - [[```692584d6```](https://github.com/jina-ai/docarray/commit/692584d6b8a2b9c1f1d6a869ecf7a0114e7e6c5c)] __-__ simplify find batched (#1598) (*Joan Fontanals*)
 - [[```91350882```](https://github.com/jina-ai/docarray/commit/91350882817cc6ed0f24aa02a6f14e7fe182fb9c)] __-__ __version__: the next version will be 0.32.2 (*Jina Dev Bot*)

<a name=release-note-0-34-0></a>
## Release Note (`0.34.0`)

> Release time: 2023-06-21 08:15:43



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  Johannes Messner,  Saba Sturua,  samsja,  maxwelljin,  Shukri,  Nikolas Pitsillos,  Joan Fontanals Martinez,  maxwelljin2,  Kacper Łukawski,  Aman Agarwal,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```eb3f8570```](https://github.com/jina-ai/docarray/commit/eb3f8570da5b1e23e21e3fe50ab0a30f136f7940)] __-__ tensor type for protobuf deserialization (#1645) (*Johannes Messner*)
 - [[```a6fdd80c```](https://github.com/jina-ai/docarray/commit/a6fdd80c69d8c23660113ad240d82167448e39f6)] __-__ sub-document support for indexer (*maxwelljin2*)
 - [[```78892703```](https://github.com/jina-ai/docarray/commit/788927034da7efc734c2cbc23ba6854dd245c3cb)] __-__ contain func for qdrant (*maxwelljin2*)
 - [[```74a683c0```](https://github.com/jina-ai/docarray/commit/74a683c04b07872646ccd6f067ae82f44ea7e370)] __-__ contain func for weaviate (*maxwelljin2*)
 - [[```6ca3aa6e```](https://github.com/jina-ai/docarray/commit/6ca3aa6eb70afc9f23b69ecf1b75b760d43614fa)] __-__ contain func for elastic (*maxwelljin2*)
 - [[```66b0f716```](https://github.com/jina-ai/docarray/commit/66b0f716a3e3cf92efe40a4346a2ccaf49897a0e)] __-__ check contain in indexer (*maxwelljin2*)
 - [[```2c123535```](https://github.com/jina-ai/docarray/commit/2c123535c2d150c6f120aad4d58df3cc6798a1c4)] __-__ support subindex on ExactNNSearch (#1617) (*maxwelljin*)

### 🐞 Bug fixes

 - [[```c3c8061f```](https://github.com/jina-ai/docarray/commit/c3c8061f3e22e50fb08404b254660006802f42a0)] __-__ docvec equality if tensors are involved (#1663) (*Johannes Messner*)
 - [[```0c27fef6```](https://github.com/jina-ai/docarray/commit/0c27fef603970e22dc1010fd2b18aa0af834ef9e)] __-__ bugs when serialize union type (#1655) (*maxwelljin*)
 - [[```dc96e38a```](https://github.com/jina-ai/docarray/commit/dc96e38a0446d36bb6c7d6f88a9209265032bb3c)] __-__ pass limit as integer (#1657) (*Joan Fontanals*)
 - [[```7e211a94```](https://github.com/jina-ai/docarray/commit/7e211a940e4a390b059ee9de5acb4afa78c93909)] __-__ pass limit as integer (#1656) (*Joan Fontanals*)
 - [[```c3db7553```](https://github.com/jina-ai/docarray/commit/c3db75538bb9d70e35b249100b6c9c7372804e4b)] __-__ update text search to match client&#39;s new sig (#1654) (*Shukri*)
 - [[```4e7e262a```](https://github.com/jina-ai/docarray/commit/4e7e262ab7394becab33cb688a066c6c62dae79c)] __-__ doc vec equality (#1641) (*Nikolas Pitsillos*)
 - [[```eae44954```](https://github.com/jina-ai/docarray/commit/eae449542c41ef39b853fa1bd3d51ebd77f56e10)] __-__ default column config should be DBConfig and not RuntimeConfig (#1648) (*Joan Fontanals*)
 - [[```d13c8c45```](https://github.com/jina-ai/docarray/commit/d13c8c450fadf4e5e3094a5b6c843e83c68734a4)] __-__ move default_column_config to DBConfig (*Joan Fontanals Martinez*)
 - [[```cd3efc6f```](https://github.com/jina-ai/docarray/commit/cd3efc6fbe68f23d1b961ce95f6d62bb26dc8141)] __-__ summary of legacy document (*maxwelljin*)
 - [[```c13739b8```](https://github.com/jina-ai/docarray/commit/c13739b80532fbcbb1b8257a5e21f08976af160b)] __-__ remove get documents method (*maxwelljin2*)
 - [[```7c807d4f```](https://github.com/jina-ai/docarray/commit/7c807d4fa8224e1fa90548bbc5e2f44907031d80)] __-__ remove get all documents method (*maxwelljin2*)
 - [[```00794486```](https://github.com/jina-ai/docarray/commit/00794486336b5bc7a852970b085e11d497de057f)] __-__ mypy issues (*maxwelljin2*)
 - [[```c8356813```](https://github.com/jina-ai/docarray/commit/c8356813acc81c5b4ce591d9ce2345b713081b08)] __-__ protobuf (de)ser for docvec (#1639) (*Johannes Messner*)
 - [[```f36c6211```](https://github.com/jina-ai/docarray/commit/f36c621104f64d4e88aeb6673e1a8ba34c3472d1)] __-__ find_and_filter for inmemory (#1642) (*Saba Sturua*)
 - [[```1abdfce0```](https://github.com/jina-ai/docarray/commit/1abdfce0eca9230bc6f75759c6279f224be23ade)] __-__ legacy document issues (*maxwelljin2*)
 - [[```b856b0b3```](https://github.com/jina-ai/docarray/commit/b856b0b3f4ccda505acc092bebecfaa00ac3fd83)] __-__ __qdrant__: working with external Qdrant collections #1630 (#1632) (*Kacper Łukawski*)
 - [[```693f877d```](https://github.com/jina-ai/docarray/commit/693f877d7e1e5921ec69e7dbb4a41f984a14d46d)] __-__ DocList and DocVec are now coerced to each other correctly (#1568) (*Aman Agarwal*)
 - [[```65afa9a1```](https://github.com/jina-ai/docarray/commit/65afa9a14c6075238aeb95f62620c93ae46aa9ca)] __-__ fix update with tensors (#1628) (*Joan Fontanals*)

### 🧼 Code Refactoring

 - [[```69dc861b```](https://github.com/jina-ai/docarray/commit/69dc861bf857c4f54d4ffc66da5160d570b4bb54)] __-__ implementation of InMemoryExactNNIndex follows DBConfig way (#1649) (*Joan Fontanals*)

### 📗 Documentation

 - [[```4e6bf49b```](https://github.com/jina-ai/docarray/commit/4e6bf49b82daae82ed25d510dc6d22f9f2e5b473)] __-__ coming from langchain (#1660) (*Saba Sturua*)
 - [[```e870eb88```](https://github.com/jina-ai/docarray/commit/e870eb8824624f4690edc9a976665d791c5d1135)] __-__ enhance DocVec section (#1658) (*maxwelljin*)
 - [[```eedd83ce```](https://github.com/jina-ai/docarray/commit/eedd83ce249941493a23bc32fc862ba7353d732c)] __-__ qdrant in memory usage (#1634) (*Saba Sturua*)

### 🍹 Other Improvements

 - [[```dc7b681e```](https://github.com/jina-ai/docarray/commit/dc7b681e1701f41fb500308fbcc154f8d09e3a1f)] __-__ upgrade version to 0.34.0 (#1664) (*Joan Fontanals*)
 - [[```deb892f1```](https://github.com/jina-ai/docarray/commit/deb892f16200c1180c7a025a11a22f87d7006bec)] __-__ fix link on pypi (#1662) (*samsja*)
 - [[```7f91e217```](https://github.com/jina-ai/docarray/commit/7f91e21737eacad0d4c7aaaec2445f5eaab3a7f7)] __-__ remove useless file (#1650) (*samsja*)
 - [[```67a328f4```](https://github.com/jina-ai/docarray/commit/67a328f444777db1f36aa99dbf879e42bd28517a)] __-__ Revert &#34;fix: move default_column_config to DBConfig&#34; (*Joan Fontanals Martinez*)
 - [[```adc48180```](https://github.com/jina-ai/docarray/commit/adc481807ca02721952501479ce4f2b209c6e62c)] __-__ drop python 3.7 (#1644) (*samsja*)
 - [[```e66bf106```](https://github.com/jina-ai/docarray/commit/e66bf1060cb020023948498df4f5266c3b23324d)] __-__ __version__: the next version will be 0.33.1 (*Jina Dev Bot*)

<a name=release-note-0-35-0></a>
## Release Note (`0.35.0`)

> Release time: 2023-07-03 11:53:25



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  Johannes Messner,  Saba Sturua,  Han Xiao,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```8f25887d```](https://github.com/jina-ai/docarray/commit/8f25887d13f27338a99199ebd85462a4d6764615)] __-__ i/o for DocVec (#1562) (*Johannes Messner*)
 - [[```e0e5cd8c```](https://github.com/jina-ai/docarray/commit/e0e5cd8ceacc9da8450094f591287d597cd7b0af)] __-__ validate file formats in url (#1606) (#1669) (*Saba Sturua*)
 - [[```a7643414```](https://github.com/jina-ai/docarray/commit/a7643414da05e1f55198836646580965a49314d2)] __-__ add method to create BaseDoc from schema (#1667) (*Joan Fontanals*)

### 🐞 Bug fixes

 - [[```bcb60ca6```](https://github.com/jina-ai/docarray/commit/bcb60ca66738dc27ce04769e754133a4e9b0e173)] __-__ better error message when docvec is unusable (#1675) (*Johannes Messner*)

### 📗 Documentation

 - [[```b6eaa94c```](https://github.com/jina-ai/docarray/commit/b6eaa94cc1853c261e5a7967a3634f017fc41968)] __-__ fix a reference in readme (#1674) (*Saba Sturua*)

### 🏁 Unit Test and CICD

 - [[```b65b385d```](https://github.com/jina-ai/docarray/commit/b65b385d36d740afb5218a3de7c258617a2e51ca)] __-__ pin pydantic version (#1682) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```3f089e52```](https://github.com/jina-ai/docarray/commit/3f089e5237c84e2ada367e30820d96018a7954d0)] __-__ update version to 0.35.0 (#1684) (*Joan Fontanals*)
 - [[```3fc6ecb7```](https://github.com/jina-ai/docarray/commit/3fc6ecb71bdc0095f2c405c17492debcc3d8412d)] __-__ fix docarray v1v2 terms (#1668) (*Han Xiao*)
 - [[```f507a5f7```](https://github.com/jina-ai/docarray/commit/f507a5f72548a5235e60f15dbcee2c35930c60c1)] __-__ __version__: the next version will be 0.34.1 (*Jina Dev Bot*)

<a name=release-note-0-36-0></a>
## Release Note (`0.36.0`)

> Release time: 2023-07-18 14:43:28



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  Saba Sturua,  Aman Agarwal,  Shukri,  samsja,  Puneeth K,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```b306c80b```](https://github.com/jina-ai/docarray/commit/b306c80b334a1d1b2bc865d53d7e9733f27445f5)] __-__ add JAX as Computation Backend  (#1646) (*Aman Agarwal*)
 - [[```069aa3aa```](https://github.com/jina-ai/docarray/commit/069aa3aa2d2eae3a1a0dca574e266a33b1edf9c9)] __-__ support redis (#1550) (*Saba Sturua*)

### 🐞 Bug fixes

 - [[```15e3ed69```](https://github.com/jina-ai/docarray/commit/15e3ed6905025ba3490607eb9659c1cfe7600160)] __-__ weaviate handles lowercase index names (#1711) (*Saba Sturua*)
 - [[```c5664016```](https://github.com/jina-ai/docarray/commit/c56640160d54ccfd2e699f7f65103672cf77f32b)] __-__ slow hnsw by caching num docs (#1706) (*Saba Sturua*)
 - [[```d2e18580```](https://github.com/jina-ai/docarray/commit/d2e1858078049217b16db0d05bc6a02be3043934)] __-__ qdrant unable to see index_name (#1705) (*Saba Sturua*)
 - [[```94a479eb```](https://github.com/jina-ai/docarray/commit/94a479eb1e1bf5e5715f61767992061f61003115)] __-__ fix search in memory with AnyEmbedding (#1696) (*Joan Fontanals*)
 - [[```62ad22aa```](https://github.com/jina-ai/docarray/commit/62ad22aa8ae3617b9464b904cd33b3115d011781)] __-__ use safe_issubclass everywhere (#1691) (*Joan Fontanals*)
 - [[```f6ce2833```](https://github.com/jina-ai/docarray/commit/f6ce2833886468e03b8eafec222be7cef3fe62e2)] __-__ avoid converting doclists in the base index (#1685) (*Saba Sturua*)

### 🧼 Code Refactoring

 - [[```0ea68467```](https://github.com/jina-ai/docarray/commit/0ea6846783a1450dc92e4ce181b430f02e32df10)] __-__ contains method in the base class (#1701) (*Saba Sturua*)
 - [[```0a1da307```](https://github.com/jina-ai/docarray/commit/0a1da3071e2f7dbcd655c2243732a2a07c95f01f)] __-__ more robust method to detect duplicate index (#1651) (*Shukri*)

### 📗 Documentation

 - [[```5089bdae```](https://github.com/jina-ai/docarray/commit/5089bdaea955f77c31495535bf99da37b85edb3b)] __-__ add docs for dict() method (#1643) (*Puneeth K*)

### 🏁 Unit Test and CICD

 - [[```e0afb5e7```](https://github.com/jina-ai/docarray/commit/e0afb5e723a7a2f3a1346eec554c7183868b98e5)] __-__ do not require black for tests more (#1694) (*Joan Fontanals*)
 - [[```0dd49538```](https://github.com/jina-ai/docarray/commit/0dd4953866faff685173ac5b6871279d545b2a50)] __-__ do not require black for tests (#1693) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```ddc73e19```](https://github.com/jina-ai/docarray/commit/ddc73e19024e2c63071fc17792bcf616b6931b0a)] __-__ upgrade version in pyproject (#1712) (*Joan Fontanals*)
 - [[```528adfc8```](https://github.com/jina-ai/docarray/commit/528adfc8f3b09fc6f3b9d65b31ca256ef34a819f)] __-__ upgrade version to 0.36 (#1710) (*Joan Fontanals*)
 - [[```a3f6998a```](https://github.com/jina-ai/docarray/commit/a3f6998a9427bc8d23bab1c4ddccd69dec220c8f)] __-__ remove one of the codecov badges (#1700) (*Joan Fontanals*)
 - [[```b364ae1a```](https://github.com/jina-ai/docarray/commit/b364ae1ae8daff4890d3fddde88ed4fe4c7e3a7c)] __-__ add codecov (#1699) (*Joan Fontanals*)
 - [[```64bbf14a```](https://github.com/jina-ai/docarray/commit/64bbf14a8d8854b95ec1c9f90ffa8c8b8a04515b)] __-__ add code of conduct (#1688) (*samsja*)
 - [[```d2655238```](https://github.com/jina-ai/docarray/commit/d2655238858a7838ca4787187aa9491d4a769e02)] __-__ __version__: the next version will be 0.35.1 (*Jina Dev Bot*)

<a name=release-note-0-37-0></a>
## Release Note (`0.37.0`)

> Release time: 2023-08-03 03:11:16



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  Saba Sturua,  Johannes Messner,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```31c2bb9c```](https://github.com/jina-ai/docarray/commit/31c2bb9c00c2cea9e148d112c1d6226d7f6c19b9)] __-__ add description and example to ID field of BaseDoc (#1737) (*Joan Fontanals*)
 - [[```efeab90d```](https://github.com/jina-ai/docarray/commit/efeab90d3840f94b15e7767a07be0f617cb8387c)] __-__ tensor_type for all DocVec serializations (#1679) (*Johannes Messner*)
 - [[```00e980dc```](https://github.com/jina-ai/docarray/commit/00e980dcfc3872b7b833184169a777527387016b)] __-__ filtering in hnsw (#1718) (*Saba Sturua*)
 - [[```7ad70bfc```](https://github.com/jina-ai/docarray/commit/7ad70bfc751841aee5f8747c681a259e2363cbe8)] __-__ update for inmemory index (#1724) (*Saba Sturua*)
 - [[```007f1131```](https://github.com/jina-ai/docarray/commit/007f1131844975d812a040c85cc21b6fa19366bd)] __-__ support milvus (#1681) (*Saba Sturua*)
 - [[```c96707a1```](https://github.com/jina-ai/docarray/commit/c96707a133e21b7810aa57da78fb2a49b448a41a)] __-__ InMemoryExactNNIndex pre filtering (#1713) (*Saba Sturua*)

### 🐞 Bug fixes

 - [[```d2c82d49```](https://github.com/jina-ai/docarray/commit/d2c82d49e3a92e5d5ba29d1e4ce9b31435c73f95)] __-__ tensor equals type raises exception (#1739) (*Johannes Messner*)
 - [[```87ec19f8```](https://github.com/jina-ai/docarray/commit/87ec19f83827cb2bc1c56087de3ef05d6bcd8e02)] __-__ add description and title to dynamic class (#1734) (*Joan Fontanals*)
 - [[```896c20be```](https://github.com/jina-ai/docarray/commit/896c20be0c32c9dc9136f2eea7bdbb8e5cf2da0e)] __-__ create more info from dynamic (#1733) (*Joan Fontanals*)
 - [[```0e130100```](https://github.com/jina-ai/docarray/commit/0e1301006403c59d99cd7c8ae77e6a7bef837838)] __-__ fix call to unsafe issubclass (#1731) (*Joan Fontanals*)
 - [[```4cd58500```](https://github.com/jina-ai/docarray/commit/4cd5850062af4515bc2aebd2b1727372a49867dc)] __-__ collection and index name in qdrant (#1723) (*Joan Fontanals*)
 - [[```304a4e9b```](https://github.com/jina-ai/docarray/commit/304a4e9b61a9eab7584cd3859a83663ddb3227ef)] __-__ fix deepcopy torchtensor (#1720) (*Joan Fontanals*)

### 🧼 Code Refactoring

 - [[```a643f6ad```](https://github.com/jina-ai/docarray/commit/a643f6adada89dd1e7e4ddf0f92c3e27fb51a23b)] __-__ hnswlib performance (#1727) (*Joan Fontanals*)
 - [[```19aec21a```](https://github.com/jina-ai/docarray/commit/19aec21aa043cbe3556744f871297ad9d171ba50)] __-__ do not recompute every time num_docs (#1729) (*Joan Fontanals*)

### 📗 Documentation

 - [[```7c10295c```](https://github.com/jina-ai/docarray/commit/7c10295c964df273483bb0391ceeee35f57c9b28)] __-__ make document indices self-contained (#1678) (*Saba Sturua*)

### 🏁 Unit Test and CICD

 - [[```7be038c8```](https://github.com/jina-ai/docarray/commit/7be038c8bcf48c77e45b7d2654b10b563603cd32)] __-__ refactor test to be independent (#1738) (*Joan Fontanals*)
 - [[```24c00cc8```](https://github.com/jina-ai/docarray/commit/24c00cc8b7cb85f1e2ef3dea76df3382380e5c99)] __-__ refactor hnswlib test subindex (#1732) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```77b4dc1f```](https://github.com/jina-ai/docarray/commit/77b4dc1f1c24c1552b01489725205b7ebd55311c)] __-__ update version (#1743) (*Joan Fontanals*)
 - [[```3be6f2b9```](https://github.com/jina-ai/docarray/commit/3be6f2b9eca79e154ca445524b6bf32ff5910fbc)] __-__ avoid extra debugging (#1730) (*Joan Fontanals*)
 - [[```24143a1f```](https://github.com/jina-ai/docarray/commit/24143a1f7a3fbcbf33523b983eec8efd8817ebc4)] __-__ refactor filter in hnswlib (#1728) (*Joan Fontanals*)
 - [[```410665ad```](https://github.com/jina-ai/docarray/commit/410665ad9ae59c0687c780bdfb2a65d8e8097f8a)] __-__ add JAX to README (#1722) (*Joan Fontanals*)
 - [[```2a866aea```](https://github.com/jina-ai/docarray/commit/2a866aea9f2779ea59b6ed26177c65bbaee33d3a)] __-__ add link to roadmap in readme (#1715) (*Joan Fontanals*)
 - [[```68b0c5b8```](https://github.com/jina-ai/docarray/commit/68b0c5b86c572f7f7f2fc3a4b838d7bd484ba77e)] __-__ __version__: the next version will be 0.36.1 (*Jina Dev Bot*)

<a name=release-note-0-37-1></a>
## Release Note (`0.37.1`)

> Release time: 2023-08-22 14:09:53



🙇 We'd like to thank all contributors for this new release! In particular,
 samsja,  AlaeddineAbdessalem,  TERBOUCHE Hacene,  Joan Fontanals,  Jina Dev Bot,  🙇


### 🐞 Bug fixes

 - [[```0ad18a63```](https://github.com/jina-ai/docarray/commit/0ad18a63e6bb1073c080eb3ac304bd90439e878b)] __-__ bump version (#1757) (*samsja*)
 - [[```46c5dfd0```](https://github.com/jina-ai/docarray/commit/46c5dfd0aa1f10723c99a3e9a39c099dc08710e8)] __-__ relax the schema check in update mixin (#1755) (*AlaeddineAbdessalem*)
 - [[```6c771125```](https://github.com/jina-ai/docarray/commit/6c771125e5e278c5c176a35ccc619e90098c7339)] __-__ __qdrant__: fix non-class type fields #1748 (#1752) (*TERBOUCHE Hacene*)
 - [[```adb0d014```](https://github.com/jina-ai/docarray/commit/adb0d0141032c72cf2412a8b998c78cd9a920a9e)] __-__ fix dynamic class creation with doubly nested schemas  (#1747) (*AlaeddineAbdessalem*)
 - [[```691d939e```](https://github.com/jina-ai/docarray/commit/691d939e8021a2589c5d5106ec1270179618599d)] __-__ fix readme test (#1746) (*samsja*)

### 📗 Documentation

 - [[```a39c4f98```](https://github.com/jina-ai/docarray/commit/a39c4f982331d6ef3145127797b1bcedc8e05248)] __-__ update readme (#1744) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```bd3d8f03```](https://github.com/jina-ai/docarray/commit/bd3d8f0354c56559c3ae8a30f06b566ed7945f6e)] __-__ __version__: the next version will be 0.37.1 (*Jina Dev Bot*)

<a name=release-note-0-38-0></a>
## Release Note (`0.38.0`)

> Release time: 2023-09-07 13:40:16



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  Johannes Messner,  samsja,  AlaeddineAbdessalem,  Jina Dev Bot,  🙇


### 🐞 Bug fixes

 - [[```fb174560```](https://github.com/jina-ai/docarray/commit/fb174560aad3b2d554c14cefeef940f8030dfdd2)] __-__ skip doc attributes in __annotations__ but not in __fields__ (#1777) (*Joan Fontanals*)
 - [[```3dc525f4```](https://github.com/jina-ai/docarray/commit/3dc525f46d8a8771b7f18070beae0c0758371dd6)] __-__ make DocList.to_json() return str instead of bytes (#1769) (*Johannes Messner*)
 - [[```2af8a0c6```](https://github.com/jina-ai/docarray/commit/2af8a0c60213a46f2b86e4c418bb5d3ef732051c)] __-__ casting in reduce before appending (#1758) (*AlaeddineAbdessalem*)

### 🧼 Code Refactoring

 - [[```08ca686d```](https://github.com/jina-ai/docarray/commit/08ca686dd397b103e6330cd479ad4519126f90b6)] __-__ use safe_issubclass (#1778) (*Joan Fontanals*)

### 📗 Documentation

 - [[```189ff637```](https://github.com/jina-ai/docarray/commit/189ff637e790c59dfea1af3447666b34c9fb9fdf)] __-__ explain how to set document config (#1773) (*Johannes Messner*)
 - [[```cd4854c9```](https://github.com/jina-ai/docarray/commit/cd4854c9b9e89abc5537be70a5a79d9a8ea47782)] __-__ add workaround for torch compile (#1754) (*Johannes Messner*)
 - [[```587ab5b3```](https://github.com/jina-ai/docarray/commit/587ab5b39160bdeda1b86d3c09d2296c443cd42e)] __-__ add note about pickling dynamically created doc class (#1763) (*Joan Fontanals*)
 - [[```61bf9c7a```](https://github.com/jina-ai/docarray/commit/61bf9c7a88a033e0551287fac5f1260fa4d355bf)] __-__ improve filtering docstrings (#1762) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```7ec88b46```](https://github.com/jina-ai/docarray/commit/7ec88b46e44b52d0400d1495e56387f367aabd2f)] __-__ update minor (#1781) (*Joan Fontanals*)
 - [[```cc2339db```](https://github.com/jina-ai/docarray/commit/cc2339db44626e622f3b2f354d0c5f8d8a0b20ea)] __-__ remove pydantic ref from issue template (#1767) (*samsja*)
 - [[```d5cb02fb```](https://github.com/jina-ai/docarray/commit/d5cb02fbd5cc7392fb92f30c1e7ea436507eb892)] __-__ __version__: the next version will be 0.37.2 (*Jina Dev Bot*)

<a name=release-note-0-39-0></a>
## Release Note (`0.39.0`)

> Release time: 2023-10-02 13:06:02



🙇 We'd like to thank all contributors for this new release! In particular,
 Joan Fontanals,  samsja,  lvzi,  Puneeth K,  Jina Dev Bot,  🙇


### 🆕 New Features

 - [[```83d2236a```](https://github.com/jina-ai/docarray/commit/83d2236a356bd108e686abae20a06c7fbc12899f)] __-__ enable dynamic doc with Pydantic v2 (#1795) (*Joan Fontanals*)
 - [[```2a1cc9e4```](https://github.com/jina-ai/docarray/commit/2a1cc9e4c975cef50760c8a459d4be30a4da4116)] __-__ add BaseDocWithoutId (#1803) (*samsja*)
 - [[```8fba9e45```](https://github.com/jina-ai/docarray/commit/8fba9e45f995f2d65efbea2c837f2f70dfe3e858)] __-__ remove JAC (#1791) (*Joan Fontanals*)
 - [[```715252a7```](https://github.com/jina-ai/docarray/commit/715252a72177e83cb81736106f34d0ab2960ce56)] __-__ hybrid pydantic support for both v1 and v2 (#1652) (*samsja*)

### 🐞 Bug fixes

 - [[```c2b08fa5```](https://github.com/jina-ai/docarray/commit/c2b08fa5cd9fab30fc4a1fe61d1373e943912fe7)] __-__ docstring tests with pydantic v2 (#1816) (*samsja*)
 - [[```3da3603b```](https://github.com/jina-ai/docarray/commit/3da3603b6a0f69016504fcbb2cd303d68cf764ec)] __-__ allow config extension in pydantic v2 (#1814) (*samsja*)
 - [[```4a1bc26a```](https://github.com/jina-ai/docarray/commit/4a1bc26a15dd02bbefa5607519f992a283bae975)] __-__ allow nested model dump via docvec (#1808) (*samsja*)
 - [[```26d776dd```](https://github.com/jina-ai/docarray/commit/26d776dd4b49ee17d92b9c97b0af9ff4ab5a4cdc)] __-__ validate before (#1806) (*samsja*)
 - [[```7209b784```](https://github.com/jina-ai/docarray/commit/7209b7849a2f9afd8cb98e01f80591d30ba28ec7)] __-__ fix double subscriptable error (#1800) (*Joan Fontanals*)
 - [[```2937e253```](https://github.com/jina-ai/docarray/commit/2937e253f8a946872a310b93de5c706279eb5adb)] __-__ make DocList compatible with BaseDocWithoutId (#1805) (*samsja*)
 - [[```0148e99c```](https://github.com/jina-ai/docarray/commit/0148e99c47ae9e1fc02c5bfc2ee717afa0633748)] __-__ milvus connection para missing (#1802) (*lvzi*)
 - [[```2f3b85e3```](https://github.com/jina-ai/docarray/commit/2f3b85e333446cfa9b8c4877c4ccf9ae49cae660)] __-__ raise exception when type of DocList is object (#1794) (*Puneeth K*)

### 🧼 Code Refactoring

 - [[```3718a747```](https://github.com/jina-ai/docarray/commit/3718a747c806b87e331903b73747d864ace42725)] __-__ add is_index_empty API (#1801) (*Joan Fontanals*)

### 📗 Documentation

 - [[```061bd81a```](https://github.com/jina-ai/docarray/commit/061bd81aa3307f13281c1c51b0ef79761cd7c5a5)] __-__ fix documentation for pydantic v2 (#1815) (*samsja*)
 - [[```d0b99909```](https://github.com/jina-ai/docarray/commit/d0b99909052d1640670764015e4e385319fdc776)] __-__ adding field descriptions to predefined mesh 3D document (#1789) (*Puneeth K*)
 - [[```18d3afce```](https://github.com/jina-ai/docarray/commit/18d3afceb1a9206c1b6f9184e7d720f4f09510c9)] __-__ adding field descriptions to predefined point cloud 3D document (#1792) (*Puneeth K*)
 - [[```4ef49394```](https://github.com/jina-ai/docarray/commit/4ef493943d1ee73613b51800980239a30fe5ae73)] __-__ adding field descriptions to predefined video document (#1775) (*Puneeth K*)
 - [[```68cc1423```](https://github.com/jina-ai/docarray/commit/68cc1423058c18d27f13cae3bd307fba5d6e9aaa)] __-__ adding field descriptions to predefined text document (#1770) (*Puneeth K*)
 - [[```441db26d```](https://github.com/jina-ai/docarray/commit/441db26dc3e58d72b44595131b42aa411c98774d)] __-__ adding field descriptions to predefined image document (#1772) (*Puneeth K*)
 - [[```35d2138c```](https://github.com/jina-ai/docarray/commit/35d2138c2d7d246b6de87de829dd870d05fc54bf)] __-__ adding field descriptions to predefined audio document (#1774) (*Puneeth K*)

### 🏁 Unit Test and CICD

 - [[```9a6b1e64```](https://github.com/jina-ai/docarray/commit/9a6b1e646e1ce5c97b81413f83b0b5a12a2c4732)] __-__ move the pydantic check inside test (#1812) (*Joan Fontanals*)
 - [[```92de15e6```](https://github.com/jina-ai/docarray/commit/92de15e6a6b42d4afe34b744e0c4f1ae581861bc)] __-__ remove skip of s3 (#1811) (*Joan Fontanals*)
 - [[```bfac0939```](https://github.com/jina-ai/docarray/commit/bfac09399a514073bcde6c72535942623ecc0e84)] __-__ remove skips (#1809) (*Joan Fontanals*)
 - [[```dce39075```](https://github.com/jina-ai/docarray/commit/dce3907560b987b9c5fc956c7ccf0cf38405d798)] __-__ fix test (#1807) (*Joan Fontanals*)
 - [[```8f32866e```](https://github.com/jina-ai/docarray/commit/8f32866e1cd3aa139a88cfca2beaa502656dc76b)] __-__ remove skipif for pydantic (#1796) (*Joan Fontanals*)

### 🍹 Other Improvements

 - [[```7693cf7c```](https://github.com/jina-ai/docarray/commit/7693cf7c27b94d616bae37a183cc9c734bc285ee)] __-__ update version to 0.39.0 (#1818) (*Joan Fontanals*)
 - [[```a4fdb77d```](https://github.com/jina-ai/docarray/commit/a4fdb77db92af2e49b8a9680950439f9ca5c1870)] __-__ fix failing test (#1793) (*Joan Fontanals*)
 - [[```805a9825```](https://github.com/jina-ai/docarray/commit/805a9825fd59848bb205461e9da71934395c0768)] __-__ __version__: the next version will be 0.38.1 (*Jina Dev Bot*)

