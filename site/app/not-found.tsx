import Link from "next/link";

export default function NotFound() {
  return (
    <section className="hero">
      <div className="wrap">
        <span className="eyebrow">404</span>
        <h1>Page not found</h1>
        <Link href="/" className="tag olive">Home →</Link>
      </div>
    </section>
  );
}
