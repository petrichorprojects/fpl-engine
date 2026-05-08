"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Users, ArrowLeftRight, Trophy,
  Calendar, MessageSquare, FlaskConical, Settings, Zap
} from "lucide-react";
import { clsx } from "clsx";

const nav = [
  { href: "/",           label: "Dashboard",   icon: LayoutDashboard },
  { href: "/squad",      label: "My Squad",    icon: Trophy          },
  { href: "/players",    label: "Players",     icon: Users           },
  { href: "/transfers",  label: "Transfers",   icon: ArrowLeftRight  },
  { href: "/rivals",     label: "Rivals",      icon: FlaskConical    },
  { href: "/calendar",   label: "Calendar",    icon: Calendar        },
  { href: "/pressers",   label: "Pressers",    icon: MessageSquare   },
  { href: "/backtest",   label: "Backtest",    icon: Zap             },
];

export function Sidebar() {
  const path = usePathname();

  return (
    <aside className="w-56 flex-shrink-0 bg-fpl-card border-r border-fpl-border flex flex-col">
      {/* Logo */}
      <div className="p-5 border-b border-fpl-border">
        <h1 className="text-fpl-green font-bold text-lg tracking-tight">
          ⚡ FPL Engine
        </h1>
        <p className="text-xs text-gray-500 mt-0.5">Analytics · v0.1</p>
      </div>

      {/* Nav */}
      <nav className="flex-1 p-3 space-y-0.5">
        {nav.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={clsx(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all",
              path === href
                ? "bg-fpl-green/10 text-fpl-green"
                : "text-gray-400 hover:text-white hover:bg-white/5"
            )}
          >
            <Icon size={16} />
            {label}
          </Link>
        ))}
      </nav>

      {/* Bottom */}
      <div className="p-3 border-t border-fpl-border">
        <Link
          href="/settings"
          className="flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm text-gray-500 hover:text-white hover:bg-white/5 transition-all"
        >
          <Settings size={16} />
          Settings
        </Link>
      </div>
    </aside>
  );
}
