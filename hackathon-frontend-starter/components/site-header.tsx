"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview" },
  { href: "/evidence", label: "Evidence" },
];

/**
 * Translucent, hairline-bottomed bar. Deference: it names the tool and gets
 * out of the way — no logo lockups, status pills or marquee strips competing
 * with the data below it.
 */
export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-30 border-b border-hairline bg-bg/80 backdrop-blur-xl">
      <div className="mx-auto flex w-full max-w-5xl items-center justify-between gap-6 px-4 py-3 sm:px-6">
        <Link href="/" className="min-w-0">
          <span className="t-headline block truncate">PAIMANA</span>
          <span className="t-caption hidden text-fg-tertiary sm:block">
            MoSPI · Infrastructure &amp; Project Monitoring
          </span>
        </Link>

        <nav className="flex shrink-0 gap-1 rounded-[var(--radius-chip)] bg-surface-sunken p-1 ring-1 ring-hairline">
          {NAV.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "t-caption rounded-[var(--radius-chip)] px-3 py-1.5 font-medium transition",
                  active ? "bg-surface text-fg shadow-sm" : "text-fg-secondary hover:text-fg",
                )}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}
