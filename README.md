﻿# calibre-metadata-Labirint


## Что это?
-----------

Плагин метаданных для [Calibre](https://github.com/kovidgoyal/calibre), позволяющий вести поиск и обновление метаданных
и обложки по магазину [labirint.ru](https://www.labirint.ru/).

> [!CAUTION]
> Данные будут найдены, если книга есть в магазине - это не глобальный поиск!



## Установка

### вариант для пользователя

1. Скачайте [ZIP-архив](https://github.com/ykmn/calibre-metadata-labirint/releases/download/0.1.16/calibre-metadata-labirint-0.1.17.zip)

2. Добавьте его в плагины Calibre (**Параметры - Плагины - Загрузить плагин из файла**)

### вариант для разработчика

```
cd ~\Projects
git clone https://github.com/ykmn/calibre-metadata-Labirint
calibre-customize -b ~\Projects\calibre-metadata-Labirint\.
```



## Данные

Плагин получает и сохраняет следующие данные:

  * ISBN
  * Labirint ID (артикул книги в магазине)
  * авторы и название книги
  * аннотация
  * серия
  * обложка
  * издательство


  > [!TIP]
  > Выберите источник метаданных в **Настройки - Загрузка метаданных - Labirint Books**.

  > [!TIP]
  > Если нужно повторно найти и скачать данные или обложку, очистите
поле **"Идентификаторы"** (где ISBN) в карточке книги.

## Скриншоты

### Загрузить метаданные

![](images/metadata-search.png)

### Загрузить обложку

![](images/metadata-cover.png)

### Карточка книги

![](images/metadata-results.png)

[![BuyMeCoffee][buymecoffeebadge]][buymecoffee]

## Версии

v0.1.16 - 2025-06-24 - Первая публичная версия.
v0.1.17 - 2025-06-25 - Добавлен парсинг жанра со страницы книги.

[buymecoffee]: https://www.tbank.ru/cf/58Tlv2umMfW
[buymecoffeebadge]: https://img.shields.io/badge/buy%20me%20a%20coffee-donate-blue.svg?style=for-the-badge
