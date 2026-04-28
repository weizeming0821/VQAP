# -*- coding: utf-8 -*-
"""统一数据生成 CLI 调度层。"""

import sys

from . import collection
from . import launcher


EXECUTION_MODE_OPTION = '--execution_mode'
EXECUTION_MODES = {'auto', 'single', 'multi'}


def _wants_combined_help(argv):
    wants_help = any(token in {'-h', '--help'} for token in argv)
    if not wants_help:
        return False

    for token in argv:
        if token == '--displays' or token.startswith('--displays='):
            return False
        if token == EXECUTION_MODE_OPTION or token.startswith(EXECUTION_MODE_OPTION + '='):
            return False
    return True


def _print_combined_help():
    print('统一数据生成入口')
    print('')
    print('模式选择规则:')
    print('  auto: 若传入 --displays，则进入多 DISPLAY launcher；否则进入单机采集。')
    print('  single: 强制单机采集模式。')
    print('  multi: 强制多 DISPLAY launcher 模式。')
    print('')
    print('常用示例:')
    print('  python scripts/generate_segmented_dataset.py --output_path ./demos_subphase')
    print('  python scripts/generate_segmented_dataset.py --displays :99.0 :99.1 --output_path ./demos_multi_gpu')
    print('  python scripts/generate_segmented_dataset.py --execution_mode single --help')
    print('  python scripts/generate_segmented_dataset.py --execution_mode multi --help')
    print('')
    print(f'可选顶层参数: {EXECUTION_MODE_OPTION} {{{", ".join(sorted(EXECUTION_MODES))}}}')
    print('')
    print('----- Single Mode Options -----')
    print(collection.build_parser().format_help().rstrip())
    print('')
    print('----- Multi Mode Options -----')
    print(launcher.build_parser().format_help().rstrip())


def _extract_execution_mode(argv):
    cleaned = []
    mode = 'auto'
    skip_next = False

    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if token == EXECUTION_MODE_OPTION:
            if index + 1 >= len(argv):
                raise ValueError('--execution_mode requires a value')
            candidate = argv[index + 1].strip().lower()
            if candidate not in EXECUTION_MODES:
                raise ValueError(
                    '--execution_mode must be one of: ' + ', '.join(sorted(EXECUTION_MODES)))
            mode = candidate
            skip_next = True
            continue

        if token.startswith(EXECUTION_MODE_OPTION + '='):
            candidate = token.split('=', 1)[1].strip().lower()
            if candidate not in EXECUTION_MODES:
                raise ValueError(
                    '--execution_mode must be one of: ' + ', '.join(sorted(EXECUTION_MODES)))
            mode = candidate
            continue

        cleaned.append(token)

    return mode, cleaned


def _has_multi_display_options(argv):
    for token in argv:
        if token == '--displays' or token.startswith('--displays='):
            return True
    return False


def resolve_execution_mode(argv):
    requested_mode, cleaned_argv = _extract_execution_mode(list(argv))
    if requested_mode == 'auto':
        resolved_mode = 'multi' if _has_multi_display_options(cleaned_argv) else 'single'
    else:
        resolved_mode = requested_mode
    return resolved_mode, cleaned_argv


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if _wants_combined_help(argv):
        _print_combined_help()
        return 0
    execution_mode, cleaned_argv = resolve_execution_mode(argv)
    if execution_mode == 'multi':
        return launcher.main(cleaned_argv)
    return collection.main(cleaned_argv)


if __name__ == '__main__':
    raise SystemExit(main())
