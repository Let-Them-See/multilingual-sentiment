import type { Metadata } from "next";
import "../src/styles/globals.css";
import Sidebar from "../src/components/layout/Sidebar";

export const metadata: Metadata = {
  title: "Multilingual Indian Sentiment",
  description:
    "Production-grade LoRA fine-tuned sentiment analysis for Hindi, Tamil, Bengali, Telugu, Marathi & code-mix.",
  keywords: ["NLP", "sentiment analysis", "India", "LoRA", "LLM", "fine-tuning"],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
        />
      </head>
      <body className="flex min-h-screen bg-mint text-teal-dark">
        <Sidebar />
        <main className="flex-1 ml-64 p-8 overflow-auto">{children}</main>
      </body>
    </html>
  );
}
