# Поставочные чекпоинты

Оба файла хранятся непосредственно в Git, без Git LFS, и приходят с `git clone`/`git pull`.
CPU и GPU Compose подключают этот каталог только для чтения; исходные данные и lab для выдачи не нужны.
Остальные веса и кэш остаются вне Git. Байты этих файлов и SHA256 рецептов не изменены.

| Файл | Модель | Размер, байт | SHA256 |
| --- | --- | ---: | --- |
| `dense-student-v3.cbm` | TabPFN dense-student-v3 (CatBoost, CPU) | 3 081 572 | `5cea6bf3e187a26ec42716ea1a91a1e5ddf50f4984ff03705b36527faf8baa3f` |
| `tabpfn-v2-regressor.ckpt` | TabPFN v2 regressor, рецепт 030 | 44 390 977 | `2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736` |

Built with PriorLabs-TabPFN. [Prior Labs License 1.1](../recipes/tabpfn-030/LICENSE.txt)
сохраняется для checkpoint и дистиллированной модели. Исходный checkpoint TabPFN не изменён;
[закреплённый источник](https://huggingface.co/Prior-Labs/TabPFN-v2-reg/blob/4972a65a1b30806315c6f92499959ffbfc69a673/tabpfn-v2-regressor.ckpt).
CPU-модель обучена в [dense_student](../lab/dense_student/README.md),
источник — `ml/lab/artifacts/dense_student_20260927_v3/bundle/student.cbm`.
Имена файлов и технические ID существующих рецептов сохранены.

[Запуск CPU/GPU и проверка](../docs/HANDOFF.md).
