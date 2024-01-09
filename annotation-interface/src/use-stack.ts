import { useCallback, useState } from "react";

export function useStack<T>(size: number) {
  const [stack, setStack] = useState<T[]>([]);

  const top = stack[stack.length - 1];

  const push = useCallback(
    (item: T) => {
      setStack((stack) => {
        const newStack = [...stack, item];
        if (newStack.length > size) {
          newStack.shift();
        }
        return newStack;
      });
    },
    [size]
  );

  const pop = useCallback(() => {
    setStack((stack) => stack.slice(0, stack.length - 1));
  }, []);

  const clear = useCallback(() => {
    setStack([]);
  }, []);

  return { push, pop, clear, top };
}
