# qeeble

## 『qeeble』とは？

- Deep Belief Network(DBN) の簡易なJava実装です。
- SUGOMORIさんのDBN実装を、自身の勉強のためにリファクタリングした結果です。他の方の参考になればと思い、公開しています。
  - もともとのリファクタリング結果からだいぶ見た目が変わってしまったので、最新のものともともとのものを別プロジェクトにしました。
- Eclipseのプロジェクトフォルダをそのままを公開していますので、ローカルに落としてEclipseにインポートしてもらえれば試せます。
- qeebleの名前の由来？　……テキトーにつけました。検索しても誰も使って無さそうな、架空の単語です。

---

## プロジェクトフォルダの説明

- qeeble-common
  - 共通的なクラス（Vector、Matrix、Modelなど）を集めたプロジェクトです。
  - いちおうインターフェイス定義していますが、今のところDense系しか実装していません（sparse系実装が必要かどうかは不明）。
- qeeble-DBN
  - DBNの簡易な実装です。
  - Restricted Boltzmann Machineは、CD-1法のみ実装を残しました（CD-k入れると見通し悪かっので…）。
  - 最終層はLogistic Regressionでfinetuningしています。
- qeeble-example
  - DBNを実行するときのmainメソッドの例です。
- qeeble-old
  - もともとの実装を単にJavaに置き換えた結果です。VectorやMatrixではなく、double配列での実装が残っています。
  - もともとの実装方針の痕跡が残っています（CD-kとか）。

---

## リンク

- [SUGOMORIさんのDeepLearning実装](https://github.com/yusugomori/DeepLearning)



