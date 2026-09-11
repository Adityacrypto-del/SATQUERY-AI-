import type { HTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('rounded-xl border border-white/[.12] bg-[#090909] shadow-2xl shadow-black/20 backdrop-blur-sm', className)} {...props} />;
}
