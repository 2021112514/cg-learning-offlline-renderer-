import re
import xml.sax
from typing import Dict, Callable, List, Tuple, Any, Optional, TypeVar, Type
from xml.sax.xmlreader import AttributesImpl


constructors: Dict[str, Callable] = {}
T = TypeVar('T')
def tag_constructor(tag_name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
    def class_decorator(cls):
        if tag_name is not None:
            constructors[tag_name] = cls.construct_tag
        else:
            constructors[cls.__name__] = cls.construct_tag
        return cls
    return class_decorator


def tag(tag_name: Optional[str] = None) -> Callable[[Type[T]], Type[T]]:
    def class_decorator(cls):
        if tag_name is not None:
            constructors[tag_name] = cls
        else:
            constructors[cls.__name__] = cls
        return cls
    return class_decorator


class XMLHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.args_stack: List[List[Tuple[str, Any]]] = []
        self.tag_stack: List[str] = []
        self.depth: int
        self.root: Any

    def startElement(self, name: str, attrs: AttributesImpl) -> None:
        self.args_stack.append([])
        self.tag_stack.append(name)
        self.depth = len(self.tag_stack)

    def endElement(self, name: str) -> None:
        args = self.args_stack.pop()
        is_leaf = self.depth == len(self.tag_stack)
        if not is_leaf:
            if name in constructors.keys():
                param = {}
                first = {}
                for k, v in args:
                    k = self.parse_arg_name(k)
                    if k not in param.keys():
                        param[k] = v
                        first[k] = True
                    elif first[k]:
                        first[k] = False
                        param[k] = [param[k], v]
                    else:
                        param[k] = param[k] + [v]
                obj = constructors[name](**param)
            else:
                obj = [v for _, v in args]
            if len(self.args_stack) == 0:
                self.root = obj
                return
            self.args_stack[-1].append((name, obj))
        self.tag_stack.pop()

    def characters(self, content: str) -> None:
        content = content.strip()
        if len(content) > 0:
            self.args_stack[-2].append((self.tag_stack[-1], self.parse_arg(content)))

    def parse_arg(self, arg: str) -> int | float | tuple | str:
        def parse_number(s: str) -> int | float | str:
            ret = s
            try:
                if '.' in s or 'e' in s:
                    ret = float(s)
                else:
                    ret = int(s)
            except:
                pass
            return ret
        if ',' in arg:
            nums = arg.removeprefix('(').removesuffix(')').split(',')
            tup = tuple(parse_number(num) for num in nums)
            has_str = False
            for num in tup:
                if isinstance(num, str):
                    has_str = True
                    break
            if not has_str:
                return tup
        return parse_number(arg)

    def parse_arg_name(self, name: str) -> str:
        def repl(s: re.Match[str]) -> str:
            group = s.group()
            return f'{group[0].lower()}_{group[1:].lower()}'
        return re.sub(r'[^^][A-Z][a-z]|[^^][A-Z]+$', repl, name).lower()