import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Sentinel-RAG | Zero-Trust AI Cybersecurity Firewall",
  description: "Real-time intra-generation semantic entropy monitoring firewall to catch and kill AI hallucinations for Celonis & IBM Granite.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} dark h-full antialiased`}
    >
      <body className="min-h-full bg-[#080a0c] text-gray-100 font-sans flex flex-col selection:bg-rose-500/30 selection:text-rose-200">
        {children}
      </body>
    </html>
  );
}
