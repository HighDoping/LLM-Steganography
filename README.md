# LLMSteganography

Steganography with the help of LLMs.  

## Introduction

This project is a steganography tool that uses LLMs to hide information in text. The tool is designed to be used with OpenAI compatible `/v1/completions` APIs. Since a huge amount of tokens are consumed, it's better to be used with a self-hosted model.  

## Usage

To encode the hidden text:

```bash
python main.py encode text_to_hide.txt output_text.txt --starter "说来话长，" --baseurl http://localhost:1234/v1 --apikey lm-studio --model qwen2.5-0.5b
```

The encoding process might be stagnant. Some retries are needed.

To decode the hidden text:

```bash
python main.py decode output_text.txt result.txt 
```

Parameters:

- `--starter`  to set the starter text. The starter text is used as the beginning of the output text.

- `--password` to set the password for the hidden text. The password is used to encrypt the hidden text.

- `--base` to set the internal encoding base. Higher base shortens the text needed for encoding but with slower speed(less chance). Default: 16

- `--char_per_index` to set how many characters are used for encoding one bit. Lower number can shorten the output text, but with higher chances that LLM stagnates at the same token. Default: 8

- `--baseurl` to set the base URL of the API.

- `--apikey`  to set the API key.

- `--model` to set the model name for the API.

## Example

All examples are created using qwen2.5-0.5b model with LM Studio running on M4 Mac Mini. Currently, using Chinese as the output language has better performance.

Encrypt plaintext `A` with password `password`, starter text is `说来话长，`, takes about 11 minutes.

Output:

```text
说来话长，那是一个周末的晚自习时间，小明在玩电脑游戏。突然，一道数学题从他脑海中闪过：“4除以0怎么算？”这时，小明感到自言自语：是0啊！就当它只是个谜语吧。\n小明很快完成了解题任务，但是他的问题并未得到全解。接下来就是下一次的上自考，这次他准备了三个备选答案，请你判断一下谁才是正确的呢？\n【分析】小明对这个数学题目不会做，也没有精力去思考它究竟要解决什么。在参考书上的答出“是0啊”，于是一味地听从自己父母的话，“这是考试题呀”，所以就将答案写成了“0”。他将34除以5等于6余4的解法记录下来：“这道题和数学题一模一样，都只是简略重复了一次。”\n可是，小明在下一次上自考时却问过一位数学老师：那我们该怎么办？这位老师告诉他：既然这个题目是考数学题啊！那么你只要计算一下4除以0的结果。如1、2、3……就都会正确。而答案“是0啊”的就是这道问题的答案了。\n小明恍然大悟，终于得到了正解，他也成为了第一个人解决了这个看似简单的题目。他仔细想了想：对啊，“有谁会说我不是数学老师呀”？于是就将这个问法挂在了前面的题上：“1、2、3……？”\n“没听说过，我来吧！我也是第一次碰到这个问题呢！”小明开始思考问题的解答过程。\n小明发现了一个错误的解法：他写的是4÷5=0.8，实际上应该是4÷0.5=8。然而在第二次上自考时，小明却将这个答出：“是0啊”！于是小明又去问那数学老师了。\n小明告诉老师他的错误方法：2/1 3/2 4/3………，因此他得出了答案“不是零”。\n可是老师告诉他：这道题还有一点特别的啊！那就是题目里有加法啦？然后他又在上面写上了“加法”几个字。最后他算出了答案：“4+1=5”，但是结果却是7而不是0。于是，老师又让小明做一遍这个题目。\n这一次，小明终于解出来：32÷6=5余2（四舍五入）…所以小明最终的答成了“是零”的答对了。\n他看着答案说：“我太笨了”！\n然而，老师又问他为什么最后不会写错呢？因为这道题目里也含有乘法啦！\n【分析】原来老师教的小明已经学过两层运算：1、2、3……和加法和乘法。因此，在做小题时，如果加上一个“0”就可以不计算了，但如果加上“×5”或者“÷6”，就会产生成立不定方程。所以答案的个位数就很容易确定。\n于是他再次解出：4+2=6，而这个结果就是正确答案了。\n小明的错解是错在对乘法不熟悉，在上自考时，他就忘记加法了。因此老师让小明进行计算时，就把“0”当作加数和因数，将1、2、3…也当做加数来算了，最后的4+6就等于10，但因为这道题没加乘号，在解答过之后就被写成了零。\n于是小明才想到了老师教的东东：在计算时可以去掉一个“0”！\n原来小明根本不知道为什么会出错。而他之所以这道题没有答对，就因为他没明白这一道理。\n从此以之为教育的启示吧，生活中的很多数理问题也都是这样，不搞清楚数学原意，又会一知半解，得出了错误的结\n【答案】7．\n由“4÷5=0.8”可得：\n所以小明的答案是7。\n分析：\n如果将除数5改成2，则6可以分成3和3。所以算式变为\n(1)若被减数、减数同时乘以10，那计算结果就变了，变成3×10=30；4÷(3÷10)=10。\n(2)如果要从商的最高位起数出余数，则必须使差中每个数字都大于或等于6。所以算式变为：\n故答案为：7．\n【答\n因为小明只在做一道题时，没搞清楚数学问题到底是什么意思；所以
```

Encode plaintext `Secret` with no password, starter text is `说来话长，`, takes about 1 minute.

Output:

```text
说来话长，关于我出生后的第2年，有一个关于我“我”不记得的“小人书”，它有三个故事：1.我是一个小坏蛋；2.妈妈在洗衣盆里找零钱时被狗咬伤了；3.妈妈没带钱去商场买玩具，可是在家拿过几个发下来的时候，妈一看就差9块了。后来这个“故事”就变成了“我记住了很多事，但少了
```

## Acknowledgements

Inspired by [抽象emoji加密器](https://bgm.tv/group/topic/414391) and [Badness 0 (Apostrophe‛s version)](https://www.youtube.com/watch?v=Y65FRxE7uMc)

Contributions appreciated.
