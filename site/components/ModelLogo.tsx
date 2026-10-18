import Image from "next/image";
import { modelFamily } from "@/lib/model-family";

const LOGOS = {
  anthropic: "claude", google: "gemini", openai: "openai",
  moonshot: "kimi", deepseek: "deepseek", xai: "grok", ibm: "granite",
};

export function ModelLogo({ model }: { model: string }) {
  return <Image className="model-logo" src={`/model-logos/${LOGOS[modelFamily(model)]}.svg`} width={20} height={20} alt="" aria-hidden="true" unoptimized />;
}
