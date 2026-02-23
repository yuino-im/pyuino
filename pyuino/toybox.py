import argparse
from .converter import YuinoConverter


def toy_run():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model_path', help='Yuino Model Path', default="YuinoLM")
    args = arg_parser.parse_args()

    print("Preparing the dictionary...")
    converter = YuinoConverter(args.model_path)

    print('--Yuino TOY-BOX-- (Exit with Ctrl+D)')
    while True:
        try:
            kana = input("かな > ")
            rsp = converter.convert(kana)
            print("漢字: " + rsp)

        except EOFError:
            print("bye!")
            break
