import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: "Veridicus — Reasoning Memory Benchmark",
  description:
    "Institutional reasoning memory layer for healthcare AI agents",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="font-sans antialiased grain-overlay">
        <Navbar />
        {children}
      </body>
    </html>
  );
}
