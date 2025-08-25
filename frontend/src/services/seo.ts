import i18n from '@/i18n'

export type RouteSeoMeta = {
  titleKey?: string
  descriptionKey?: string
  title?: string
  description?: string
}

function setMetaTag(selector: string, attr: 'content', value: string | null | undefined) {
  if (!value) return
  const el = document.querySelector(selector) as HTMLMetaElement | null
  if (el) el.setAttribute(attr, value)
}

function setOrCreateLink(rel: string, href: string, hreflang?: string) {
  let selector = `link[rel="${rel}"]`
  if (hreflang) selector += `[hreflang="${hreflang}"]`
  let el = document.querySelector(selector) as HTMLLinkElement | null
  if (!el) {
    el = document.createElement('link')
    el.setAttribute('rel', rel)
    if (hreflang) el.setAttribute('hreflang', hreflang)
    document.head.appendChild(el)
  }
  el.setAttribute('href', href)
}

export function applySeo(meta: RouteSeoMeta): void {
  const t = i18n.global.t
  const title = meta.titleKey ? (t(meta.titleKey) as string) : meta.title
  const description = meta.descriptionKey ? (t(meta.descriptionKey) as string) : meta.description

  if (title) document.title = title
  setMetaTag('meta[name="description"]', 'content', description)

  // Open Graph
  setMetaTag('meta[property="og:title"]', 'content', title)
  setMetaTag('meta[property="og:description"]', 'content', description)
  const locale = i18n.global.locale.value as string
  const ogLocale = locale === 'tr' ? 'tr_TR' : 'en_US'
  setMetaTag('meta[property="og:locale"]', 'content', ogLocale)

  // Twitter
  setMetaTag('meta[name="twitter:title"]', 'content', title)
  setMetaTag('meta[name="twitter:description"]', 'content', description)

  // Canonical and hreflang alternates using ?hl locale param
  try {
    const url = new URL(window.location.href)
    url.searchParams.delete('hl')
    const base = url.toString()

    const hrefFor = (lng: string) => {
      const u = new URL(base)
      u.searchParams.set('hl', lng)
      return u.toString()
    }

    setOrCreateLink('canonical', hrefFor(locale))
    setOrCreateLink('alternate', hrefFor('en'), 'en')
    setOrCreateLink('alternate', hrefFor('tr'), 'tr')
    setOrCreateLink('alternate', hrefFor('en'), 'x-default')
  } catch {
    // ignore
  }
}
