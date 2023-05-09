




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

