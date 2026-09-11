import type { ButtonHTMLAttributes } from 'react';
import { cn } from '../../lib/utils';

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: 'default' | 'outline' | 'ghost';
};

export function Button({ className, variant = 'default', ...props }: ButtonProps) {
  return <button className={cn('inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-white/50 disabled:pointer-events-none disabled:opacity-50', variant === 'default' && 'bg-white text-black hover:bg-neutral-200', variant === 'outline' && 'border border-white/15 bg-white/[.03] text-white hover:bg-white/[.08]', variant === 'ghost' && 'text-neutral-300 hover:bg-white/[.06] hover:text-white', className)} {...props} />;
}
