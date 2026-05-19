import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'Drug Discovery Canvas',
  description: 'Agentic in-silico compound-protein simulation with adaptive memory',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased bg-gray-950 text-gray-100 min-h-screen">
        {children}
      </body>
    </html>
  );
}
