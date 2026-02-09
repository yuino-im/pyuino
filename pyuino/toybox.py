import argparse
from .converter import YuinoConverter


def toy_run():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model_path', help='Yuino Model Path', default="YuinoLM")
    args = arg_parser.parse_args()

    converter = YuinoConverter(args.model_path)

    print('--Yuino TOY-BOX--')
    while True:
        kana = input("かな > ")
        rsp = converter.convert(kana)
        print("漢字: " + rsp)
